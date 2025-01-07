from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Set, Tuple
import gensim.downloader
from numpy import dot, ndarray
from numpy.linalg import norm
import random
from pydantic import BaseModel
import json
import uuid
from functools import lru_cache
from dataclasses import dataclass
from collections import deque
import asyncio
import time
import logging
import nltk
from nltk.corpus import words

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_response(response: dict):
    print("\n=== WebSocket Response ===")
    print(json.dumps(response, indent=2))
    print("========================\n")

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
print("Loading word model...")
model = gensim.downloader.load('glove-wiki-gigaword-100')

# Download required NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Create set of valid English words
ENGLISH_WORDS = set(words.words())
VALID_WORDS: Set[str] = {word.lower() for word in model.index_to_key 
                        if len(word) > 3 
                        and word.isalpha() 
                        and word.lower() in ENGLISH_WORDS}

DIFFICULTY_THRESHOLDS = {
    'easy': 0.6,
    'medium': 0.5,
    'hard': 0.3,
    'expert': 0.2
}

@dataclass
class GameState:
    words: List[Dict[str, str]]
    difficulty: str
    threshold: float
    game_over: bool
    timer_duration: Optional[int]
    used_words: Set[str]
    timer_task: Optional[asyncio.Task] = None
    last_move_time: Optional[float] = None

    def __init__(self, difficulty: str, timer_duration: Optional[int] = None):
        self.words = []
        self.difficulty = difficulty
        self.threshold = DIFFICULTY_THRESHOLDS[difficulty]
        self.game_over = False
        self.timer_duration = timer_duration
        self.used_words = set()
        self.timer_task = None
        self.last_move_time = None

class GameInitRequest(BaseModel):
    timerDuration: Optional[int] = None

# Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_games: Dict[str, GameState] = {}

    async def connect(self, game_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[game_id] = websocket

    def disconnect(self, game_id: str):
        self.active_connections.pop(game_id, None)
        self.active_games.pop(game_id, None)

    def get_game(self, game_id: str) -> Optional[GameState]:
        return self.active_games.get(game_id)

manager = ConnectionManager()

# Cache similarity calculations
@lru_cache(maxsize=10000)
def cosine_similarity(word1: str, word2: str) -> Optional[float]:
    try:
        vec1: ndarray = model[word1]
        vec2: ndarray = model[word2]
        return float(dot(vec1, vec2)/(norm(vec1)*norm(vec2)))
    except KeyError:
        return None

def get_similarity_table(word: str, previous_words: List[Dict[str, str]], threshold: float) -> Dict:
    similarities_list = []
    for prev in previous_words:
        if prev["word"] != word:
            sim = cosine_similarity(word, prev["word"])
            similarities_list.append({
                "previousWord": prev["word"],
                "playedBy": prev["player"],
                "similarity": round(sim, 3) if sim else None,
                "tooSimilar": sim is not None and sim > threshold
            })
    return {"forWord": word, "similarities": similarities_list}

def check_word_validity(word: str, game_state: GameState) -> Tuple[bool, Optional[str], Optional[str]]:
    # Quick validation checks
    if word in game_state.used_words:
        return False, f"'{word}' was already used.", None
    if word not in VALID_WORDS:
        return False, f"Invalid word: '{word}'", None

    # Check similarity threshold
    for prev in game_state.words:
        sim = cosine_similarity(word, prev["word"])
        if sim and sim > game_state.threshold:
            return False, f"'{word}' is too similar to '{prev['word']}'", prev["word"]

    return True, None, None

def get_computer_word(game_state: GameState) -> Optional[str]:
    # Use random sampling for better performance
    available_words = VALID_WORDS - game_state.used_words
    candidates = random.sample(available_words, min(100, len(available_words)))
    
    for candidate in candidates:
        is_valid, _, _ = check_word_validity(candidate, game_state)
        if is_valid:
            return candidate
    return None

def get_game_summary(game_state: GameState, final_word: str, final_player: str) -> Dict:
    """Calculate game statistics including the final word and winner"""
    all_words = game_state.words + [{"word": final_word, "player": final_player}]
    
    total_words = len(all_words)
    human_words = sum(1 for word in all_words if word["player"] == "human")
    computer_words = sum(1 for word in all_words if word["player"] == "computer")
    
    # Determine winner based on who made the last valid move
    winner = "computer" if final_player == "human" else "human"
    if final_player == "computer" and not computer_words:  # Computer couldn't find first word
        winner = "human"
    
    return {
        "totalWords": total_words,
        "humanWords": human_words,
        "computerWords": computer_words,
        "wordHistory": all_words,
        "winner": winner
    }

@app.get("/difficulties")
async def get_difficulties():
    return {"difficulties": list(DIFFICULTY_THRESHOLDS.keys())}

@app.post("/initialize-game/{difficulty}")
async def initialize_game(difficulty: str, request: GameInitRequest):
    if difficulty not in DIFFICULTY_THRESHOLDS:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
    
    game_id = str(uuid.uuid4())
    game_state = GameState(difficulty, request.timerDuration)
    manager.active_games[game_id] = game_state
    
    return {
        "game_id": game_id,
        "threshold": DIFFICULTY_THRESHOLDS[difficulty],
        "timerDuration": request.timerDuration
    }

async def handle_timer(websocket: WebSocket, game_state: GameState):
    try:
        while True:
            await asyncio.sleep(1)
            if game_state.last_move_time is None:
                continue
            
            elapsed = time.time() - game_state.last_move_time
            time_left = game_state.timer_duration - elapsed
            
            if time_left <= 0:
                # When timer expires, computer always wins
                game_summary = get_game_summary(game_state, "", "human")  # Pass "human" as final player so computer wins
                game_summary["winner"] = "computer"  # Force computer as winner
                
                response = {
                    "type": "gameOver",
                    "message": "Time's up! Computer wins!",
                    "gameSummary": {
                        **game_summary,
                        "totalWords": len(game_state.words),
                        "humanWords": sum(1 for w in game_state.words if w["player"] == "human"),
                        "computerWords": sum(1 for w in game_state.words if w["player"] == "computer"),
                        "message": "Time's up! Computer wins!",
                        "winner": "computer"  # Ensure computer wins
                    },
                    "wordChain": [{"word": w["word"], "player": w["player"]} for w in game_state.words]
                }
                log_response(response)
                await websocket.send_json(response)
                await websocket.send_json({"type": "gameEnd"})
                await websocket.close()
                break
            
            await websocket.send_json({
                "type": "timerUpdate",
                "timeLeft": round(time_left)
            })
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during timer for game {id(game_state)}")
    except Exception as e:
        logger.error(f"Timer error for game {id(game_state)}: {str(e)}")

async def end_game_immediately(websocket: WebSocket, message: str, game_state: GameState, final_word: str, final_player: str):
    """Helper function to handle immediate game termination"""
    game_summary = get_game_summary(game_state, final_word, final_player)
    similarity_table = get_similarity_table(final_word, game_state.words, game_state.threshold)
    
    response = {
        "type": "gameOver",
        "message": message,
        "gameSummary": {
            **game_summary,
            "message": message,
            "winner": game_summary["winner"],
            "finalWord": final_word,
            "similarityTable": similarity_table
        },
        "similarityTable": similarity_table,
        "finalWord": final_word,
        "wordChain": [{"word": w["word"], "player": w["player"]} for w in game_state.words]
    }
    log_response(response)
    await websocket.send_json(response)
    await websocket.send_json({"type": "gameEnd"})
    await websocket.close()

@app.websocket("/game/{game_id}")
async def game_websocket(websocket: WebSocket, game_id: str):
    try:
        game_state = manager.get_game(game_id)
        if not game_state:
            await websocket.close(code=4000, reason="Invalid game ID")
            return

        await manager.connect(game_id, websocket)

        if game_state.timer_duration:
            game_state.last_move_time = time.time()
            game_state.timer_task = asyncio.create_task(handle_timer(websocket, game_state))

        # Computer's first move
        computer_word = get_computer_word(game_state)
        if not computer_word:
            await end_game_immediately(
                websocket,
                "Computer couldn't find a valid starting word",
                game_state,
                "",
                "computer"
            )
            return

        game_state.words.append({"word": computer_word, "player": "computer"})
        game_state.used_words.add(computer_word)
        
        response = {
            "type": "computerMove",
            "word": computer_word,
            "moveAccepted": True,
            "similarityTable": get_similarity_table(computer_word, [], game_state.threshold),
            "isComputerTable": True,
            "wordChain": [{"word": w["word"], "player": w["player"]} for w in game_state.words]
        }
        log_response(response)
        await websocket.send_json(response)

        # Main game loop
        while not game_state.game_over:
            try:
                # Handle player's move
                data = await websocket.receive_text()
                player_word = json.loads(data)["word"].lower().strip()
                logger.info(f"Received player word: {player_word}")

                is_valid, error_msg, similar_word = check_word_validity(player_word, game_state)
                
                if not is_valid:
                    if similar_word:  # Word is too similar - game over
                        game_state.game_over = True
                        await end_game_immediately(
                            websocket,
                            error_msg,
                            game_state,
                            player_word,
                            "human"
                        )
                        return

                    # Other invalid cases - let player try again
                    response = {
                        "type": "playerMove",
                        "message": error_msg,
                        "moveDenied": True
                    }
                    log_response(response)
                    await websocket.send_json(response)
                    continue

                # Valid player move
                game_state.words.append({"word": player_word, "player": "human"})
                game_state.used_words.add(player_word)
                
                response = {
                    "type": "playerMove",
                    "moveAccepted": True,
                    "similarityTable": get_similarity_table(player_word, game_state.words[:-1], game_state.threshold),
                    "isComputerTable": False,
                    "wordChain": [{"word": w["word"], "player": w["player"]} for w in game_state.words]
                }
                log_response(response)
                await websocket.send_json(response)

                if game_state.timer_duration:
                    game_state.last_move_time = time.time()

                # Computer's next turn
                computer_word = get_computer_word(game_state)
                if not computer_word:
                    game_state.game_over = True
                    await end_game_immediately(
                        websocket,
                        "Computer exhausted all possibilities",
                        game_state,
                        player_word,
                        "human"
                    )
                    return

                is_valid, error_msg, similar_word = check_word_validity(computer_word, game_state)
                if not is_valid and similar_word:
                    game_state.game_over = True
                    await end_game_immediately(
                        websocket,
                        f"Game Over! Computer's word '{computer_word}' was too similar to '{similar_word}'",
                        game_state,
                        computer_word,
                        "computer"
                    )
                    return

                # Valid computer move
                game_state.words.append({"word": computer_word, "player": "computer"})
                game_state.used_words.add(computer_word)
                
                response = {
                    "type": "computerMove",
                    "word": computer_word,
                    "moveAccepted": True,
                    "similarityTable": get_similarity_table(computer_word, game_state.words[:-1], game_state.threshold),
                    "isComputerTable": True,
                    "wordChain": [{"word": w["word"], "player": w["player"]} for w in game_state.words]
                }
                log_response(response)
                await websocket.send_json(response)

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"Game error: {str(e)}")
        await websocket.close(code=4000, reason=str(e))
    finally:
        if game_state.timer_task:
            game_state.timer_task.cancel()
        manager.disconnect(game_id)
        logger.info(f"Game {game_id} cleaned up")