from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from typing import Dict, List, Optional
import gensim.downloader
from numpy import dot
from numpy.linalg import norm
import random
from pydantic import BaseModel
import json
import uuid
import websockets

app = FastAPI()

# Update CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - update this for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
print("Loading word model...")
model = gensim.downloader.load('glove-wiki-gigaword-100')

DIFFICULTY_THRESHOLDS = {
    'easy': 0.6,
    'medium': 0.5,
    'hard': 0.3,
    'expert': 0.2
}

class GameState:
    def __init__(self, difficulty: str, timer_duration: Optional[int] = None):
        self.words: List[Dict[str, str]] = []
        self.difficulty = difficulty
        self.threshold = DIFFICULTY_THRESHOLDS[difficulty]
        self.game_over = False
        self.timer_duration = timer_duration

class GameInitRequest(BaseModel):
    timerDuration: Optional[int] = None

# Store active games
active_games: Dict[str, GameState] = {}

def cosine_similarity(word1: str, word2: str) -> Optional[float]:
    try:
        vec1 = model[word1]
        vec2 = model[word2]
        return float(dot(vec1, vec2)/(norm(vec1)*norm(vec2)))
    except KeyError:
        return None

def is_word_valid(word: str) -> bool:
    return word in model

def get_similarity_table(word: str, previous_words: List[Dict[str, str]], threshold: float):
    return {
        "forWord": word,
        "similarities": [
            {
                "previousWord": prev["word"],
                "playedBy": prev["player"],
                "similarity": round(cosine_similarity(word, prev["word"]), 3) if cosine_similarity(word, prev["word"]) else None,
                "tooSimilar": bool(cosine_similarity(word, prev["word"]) and cosine_similarity(word, prev["word"]) > threshold)
            }
            for prev in previous_words if prev["word"] != word  # Skip the current word
        ]
    }

def check_word_validity(word: str, previous_words: List[Dict[str, str]], threshold: float):
    if any(char.isdigit() for char in word):
        return False, f"'{word}' contains numbers. Numbers are not allowed.", None

    if any(prev["word"] == word for prev in previous_words):
        return False, f"'{word}' was already used.", None

    for prev in previous_words:
        similarity = cosine_similarity(word, prev["word"])
        if similarity and similarity > threshold:
            return False, f"'{word}' is too similar to '{prev['word']}'", prev["word"]

    return True, None, None

def get_game_summary(words: List[Dict[str, str]]):
    return {
        "totalWords": len(words),
        "wordHistory": words
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
    active_games[game_id] = game_state
    
    return {
        "game_id": game_id,
        "threshold": DIFFICULTY_THRESHOLDS[difficulty],
        "timerDuration": request.timerDuration
    }

@app.websocket("/game/{game_id}")
async def game_websocket(websocket: WebSocket, game_id: str):
    print(f"Attempting to connect game: {game_id}")
    
    if game_id not in active_games:
        print(f"Game ID {game_id} not found in active games")
        await websocket.close(code=4000, reason="Invalid game ID")
        return
        
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted for game: {game_id}")
        game_state = active_games[game_id]
        
        while True:
            # Don't proceed with computer's turn if game is already over
            if game_state.game_over:
                break

            # Computer's turn
            max_attempts = 100
            computer_word = None
            
            for _ in range(max_attempts):
                candidate = random.choice(list(model.index_to_key))
                is_valid, _, _ = check_word_validity(
                    candidate, 
                    game_state.words, 
                    game_state.threshold
                )
                if is_valid:
                    computer_word = candidate
                    break
            
            if not computer_word:
                await websocket.send_json({
                    "type": "gameOver",
                    "message": "Computer couldn't find a valid word",
                    "gameSummary": get_game_summary(game_state.words),
                    "similarityTable": get_similarity_table(
                        computer_word if computer_word else "",
                        game_state.words,
                        game_state.threshold
                    )
                })
                game_state.game_over = True
                break

            # Get and send computer's move similarity table
            game_state.words.append({"word": computer_word, "player": "computer"})
            computer_similarity = get_similarity_table(
                computer_word,
                game_state.words[:-1],
                game_state.threshold
            )
            
            await websocket.send_json({
                "type": "computerMove",
                "word": computer_word,
                "similarityTable": computer_similarity,
                "isComputerTable": True
            })
            
            # Wait for player's move
            try:
                data = await websocket.receive_text()
                player_data = json.loads(data)
                player_word = player_data["word"].lower().strip()
                
                if not is_word_valid(player_word):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Word not found in vocabulary"
                    })
                    continue
                
                # Get similarity table for player's word
                player_similarity = get_similarity_table(
                    player_word,
                    game_state.words,
                    game_state.threshold
                )
                
                # Check word validity
                is_valid, error_msg, _ = check_word_validity(
                    player_word,
                    game_state.words,
                    game_state.threshold
                )
                
                if not is_valid:
                    # First send the player's final table
                    await websocket.send_json({
                        "type": "playerMove",
                        "similarityTable": player_similarity,
                        "isComputerTable": False
                    })
                    # Then send game over with final state
                    await websocket.send_json({
                        "type": "gameOver",
                        "message": error_msg,
                        "gameSummary": get_game_summary(game_state.words),
                        "similarityTable": player_similarity
                    })
                    game_state.game_over = True
                    break
                
                game_state.words.append({"word": player_word, "player": "human"})
                
                # First send player's table
                await websocket.send_json({
                    "type": "playerMove",
                    "similarityTable": player_similarity,
                    "isComputerTable": False
                })
                # Then send computer's table
                await websocket.send_json({
                    "type": "moveAccepted",
                    "similarityTable": computer_similarity,
                    "isComputerTable": True
                })
                
            except WebSocketDisconnect:
                print(f"WebSocket disconnected for game: {game_id}")
                break
            except json.JSONDecodeError:
                print(f"Invalid JSON received in game {game_id}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message format"
                })
            except Exception as e:
                print(f"Error processing player move in game {game_id}: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing move"
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for game: {game_id}")
    except Exception as e:
        print(f"Connection error in game {game_id}: {str(e)}")
        await websocket.close(code=4000, reason=str(e))
    finally:
        if game_id in active_games:
            del active_games[game_id]
        print(f"Game {game_id} cleaned up")