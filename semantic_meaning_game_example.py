import gensim.downloader
import random
from numpy import dot
from numpy.linalg import norm
from colorama import init, Fore, Style
import os

# Initialize colorama
init()

# Function to clear terminal
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Load the model
print("Loading word model...")
model = gensim.downloader.load('glove-wiki-gigaword-100')

DIFFICULTY_THRESHOLDS = {
    'easy': 0.6,    # Very similar words are needed to lose
    'medium': 0.5,  # Moderately similar words will cause loss
    'hard': 0.3,    # Even slightly similar words will cause loss
    'expert': 0.2   # Nearly any semantic relationship will cause loss
}

def get_difficulty():
    while True:
        print("\nChoose difficulty level:")
        for level in DIFFICULTY_THRESHOLDS:
            print(f"- {level}")
        choice = input("Enter difficulty: ").lower().strip()
        if choice in DIFFICULTY_THRESHOLDS:
            return choice, DIFFICULTY_THRESHOLDS[choice]
        print("Invalid choice, please try again.")

def cosine_similarity(word1, word2):
    try:
        vec1 = model[word1]
        vec2 = model[word2]
        return dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    except KeyError:
        return None

def is_word_valid(word):
    return word in model

def display_similarity_table(word, previous_words, threshold):
    """Display similarity scores between current word and previous words"""
    print(f"\n{Fore.CYAN}Similarity Table for word: '{Fore.YELLOW}{word}{Fore.CYAN}'{Style.RESET_ALL}")
    print(Fore.BLUE + "-" * 65 + Style.RESET_ALL)
    print(f"{Fore.GREEN}{'Previous Word':<20} | {'Similarity to ' + word:<30} | {'Too Similar?':<10}{Style.RESET_ALL}")
    print(Fore.BLUE + "-" * 65 + Style.RESET_ALL)
    
    for prev_word in previous_words:  # Removed slice to check all words
        similarity = cosine_similarity(word, prev_word)
        is_similar = similarity and similarity > threshold
        similarity_str = f"{similarity:.3f}" if similarity else "N/A"
        
        color = Fore.RED if is_similar else Fore.WHITE
        is_similar_text = "YES" if is_similar else "NO"
        
        print(f"{color}{prev_word:<20} | {similarity_str:<30} | {is_similar_text:<10}{Style.RESET_ALL}")
    print(Fore.BLUE + "-" * 65 + Style.RESET_ALL)

def check_word_validity(word, previous_words, threshold, player=None):  # Removed window parameter
    """
    Comprehensive check for word validity including:
    - Not a number
    - Similarity with all previous words
    - Word repetition
    Returns (is_valid, error_message, similar_word, similarity)
    """
    # Check if word contains numbers
    if any(char.isdigit() for char in word):
        return False, f"'{word}' contains numbers. Numbers are not allowed.", None, None

    # Check for repetition (now checks all previous words)
    if word in previous_words:
        return False, f"'{word}' was already used.", None, None

    # Check similarity with all previous words
    for prev_word in previous_words:  # Removed slice to check all words
        similarity = cosine_similarity(word, prev_word)
        if similarity and similarity > threshold:
            msg = f"Game Over! '{word}' is too similar to '{prev_word}'" if player else \
                  f"'{word}' is too similar to '{prev_word}' (similarity: {similarity:.3f})"
            if player:
                msg += f"\n{player} loses!"
            return False, msg, prev_word, similarity
            
    return True, None, None, None

def display_game_summary(game_words, difficulty, threshold):  # Added threshold parameter
    # First display final similarity table for the last word if there are words
    if game_words:
        display_similarity_table(game_words[-1], game_words[:-1], threshold)
    
    print(f"\n{Fore.CYAN}=== Game Summary ==={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total words played: {Fore.YELLOW}{len(game_words)}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Difficulty level: {Fore.YELLOW}{difficulty.upper()}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Word chain:{Style.RESET_ALL}")
    
    # Display word chain in columns (4 words per row)
    for i in range(0, len(game_words), 4):
        chunk = game_words[i:i+4]
        formatted_words = [f"{Fore.YELLOW}{word:<20}{Style.RESET_ALL}" for word in chunk]
        print("".join(formatted_words))

def play_game():
    clear_terminal()  # Initial clear at game start
    print(f"\n{Fore.CYAN}Welcome to the Semantic Meaning Game!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Rules: Don't use words that are similar in meaning to ANY recent word.")
    print(f"Also, you cannot repeat any word used in the last 15 turns.{Style.RESET_ALL}")
    
    difficulty, threshold = get_difficulty()
    print(f"\n{Fore.GREEN}Playing on {Fore.YELLOW}{difficulty.upper()}{Fore.GREEN} mode "
          f"(similarity threshold: {Fore.YELLOW}{threshold}{Fore.GREEN}){Style.RESET_ALL}")
    print("The computer goes first.\n")

    game_words = []
    current_player = "computer"

    while True:
        if current_player == "computer":
            # Computer's turn
            max_attempts = 100
            attempts = 0
            word = None
            
            while attempts < max_attempts:
                candidate = random.choice(list(model.index_to_key))
                is_valid, error_msg, _, _ = check_word_validity(
                    candidate, game_words, threshold, "Computer"
                )
                if is_valid:
                    word = candidate
                    break
                attempts += 1
            
            if word is None:
                print(f"{Fore.RED}Computer couldn't find a valid word. Game over!{Style.RESET_ALL}")
                display_game_summary(game_words, difficulty, threshold)
                break
            
            print(f"\n{Fore.CYAN}Computer chose: {Fore.YELLOW}{word}{Style.RESET_ALL}")
            # Always display similarity table before checking game end
            display_similarity_table(word, game_words, threshold)
            
            # Check validity and end game if needed
            is_valid, error_msg, _, _ = check_word_validity(
                word, game_words, threshold, "Computer"
            )
            if not is_valid and game_words:
                print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                display_game_summary(game_words, difficulty, threshold)
                break
                
            game_words.append(word)
            current_player = "human"
        else:
            # Human's turn
            while True:
                word = input(f"\n{Fore.CYAN}Your turn. Enter a word: {Style.RESET_ALL}").strip().lower()
                if not is_word_valid(word):
                    print(f"{Fore.RED}Word not found in vocabulary. Try another word.{Style.RESET_ALL}")
                    continue
                
                # Always display similarity table before checking game end
                display_similarity_table(word, game_words, threshold)
                
                # Check validity and end game if needed
                is_valid, error_msg, _, _ = check_word_validity(
                    word, game_words, threshold, "Human"
                )
                if not is_valid:
                    print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                    if "Game Over" in error_msg:
                        display_game_summary(game_words, difficulty, threshold)
                        return
                    continue
                break

            game_words.append(word)
            current_player = "computer"

            print(f"\n{Fore.CYAN}Current word chain:{Style.RESET_ALL}", 
                  " -> ".join(f"{Fore.YELLOW}{w}{Style.RESET_ALL}" for w in game_words))
            
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            clear_terminal()

if __name__ == "__main__":
    play_game()

