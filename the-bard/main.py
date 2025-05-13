import os
import random
import json
import time # Using a generic name for time
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    Gemma3ForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead # GRPOTrainer might use a standard model too
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize # For BLEU score
import ollama
import numpy as np

# --- Configuration ---
SFT_MODEL_DIR = "./sft_poem_model_grpo"
RL_MODEL_DIR = "./rl_poem_model_grpo"
POEM_DATASET_FILE = "poems.jsonl"
# GENERATOR_MODEL_NAME_FOR_TOKENIZER = "meta-llama/Llama-3.2-1B"
GENERATOR_MODEL_NAME_FOR_TOKENIZER = "google/gemma-3-1b-it"
# GENERATOR_MODEL_NAME_FOR_TOKENIZER = "google/gemma-3-1b-pt"
# PRETRAINED_MODEL_NAME = "meta-llama/Llama-3.2-1B"
PRETRAINED_MODEL_NAME = "google/gemma-3-1b-it"
# PRETRAINED_MODEL_NAME = "google/gemma-3-1b-pt"

GENERATOR_CONFIG = GPT2Config(
    vocab_size=50257,
    n_positions=256,
    n_embd=256,
    n_layer=3,
    n_head=4,
    bos_token_id=50256,
    eos_token_id=50256,
)

OLLAMA_JUDGE_MODEL = "gemma3"
OLLAMA_JUDGE_TEMPERATURE = 0.1 # Low temp for more consistent scoring

# SFT Training Arg
SFT_TRAINING_ARGS = TrainingArguments(
    output_dir=SFT_MODEL_DIR,
    num_train_epochs=3, # TODO: Adjust
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=42,
)

# GRPO Config
GRPO_TRAINING_ARGS = GRPOConfig(
    output_dir=RL_MODEL_DIR,
    learning_rate=1e-6,
    per_device_train_batch_size=1, # Number of prompts processed at once. Each prompt generates `num_generations` poems.
    gradient_accumulation_steps=4, # Effective batch size = per_device_train_batch_size * num_devices * gradient_accumulation_steps
    max_prompt_length=280, # Max length of the prompt
    max_completion_length=(GENERATOR_CONFIG.n_positions * 2) // 3, # Max length of the generated completion
    num_generations=2, # Number of poems to generate per prompt for group evaluation.
    num_train_epochs=10, # TODO: Number of PPO-like epochs per GRPO iteration (how many times to iterate over collected rollouts)
    beta=0.05, # KL divergence coefficient (TRL GRPOTrainer uses 'beta' for KL)
    optim="adamw_torch",
    remove_unused_columns=False,
    seed=42,
    report_to="wandb",
    temperature=1.1,
)
GRPO_ITERATIONS = 30 # TODO: Total number of GRPO training iterations (outer loop)
DIVERSITY_WEIGHT = 0.01 # TODO: Tune weight for the diversity component of the reward

# --- Helper Functions (Dataset and Tokenizer same as before) ---
def create_dummy_poem_dataset(filepath="poems.jsonl", num_poems=100):
    if not os.path.exists(filepath):
        print(f"Creating dummy poem dataset at {filepath}...")
        poems_data = [
            {"text": "The roses are red, the violets are blue,"},
            {"text": "Sugar is sweet, and so are you."},
            {"text": "In fields of green, under skies so bright,"},
            {"text": "A gentle breeze whispers through the night."},
            {"text": "Stars above, a moonlit gleam,"},
            {"text": "Lost in a wonderful, poetic dream."}
        ]
        with open(filepath, "w") as f:
            for i in range(num_poems):
                f.write(json.dumps(random.choice(poems_data)) + "\n")
        print("Dummy dataset created.")
    else:
        print(f"Using existing dataset at {filepath}")

def load_and_tokenize_dataset(tokenizer, file_path="poems.jsonl", block_size=128):
    try:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError: return None
    except json.JSONDecodeError: return None
    texts = [item['text'] for item in data if item.get('text') and len(item['text'].split()) > 2]
    if not texts: return None
    dataset = Dataset.from_dict({"text": texts})
    def tokenize_function(examples):
        return tokenizer([text + tokenizer.eos_token for text in examples["text"]],
                         truncation=True, max_length=block_size, padding="max_length")
    return dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

# --- Ollama Judge for Scoring a SINGLE Poem ---
def get_ollama_score_for_single_poem(poem_text: str, ollama_model: str, retries=2) -> float:
    """
    Asks Ollama to score a single poem on a scale of 1 to 10.
    This is a change from A/B comparison to fit GRPO's reward structure.
    """
    # TODO: Engineer this prompt to work well
    prompt = f"""You are a meticulous literary critic. Your task is to score the following poem on a scale of 1 to 10 based on creativity, coherence, emotional impact, poetic devices, and originality. Only provide a numerical score.

**CRITICAL INSTRUCTIONS:**
1.  **Meaningful Content is Required:** The text MUST contain meaningful English words forming coherent phrases or sentences.
2.  **Score for Specific Inputs:**
    * If the "Poem" text is empty, consists ONLY of whitespace (newlines, spaces), or ONLY of non-alphanumeric characters (like '_______' or '.......'), assign a **Score of 1**.
    * If the "Poem" text appears to be plagiarized or a trivial modification of a well-known work, assign a **Score of 1 to 3**
    * If the "Poem" text contains any html tags, asign a **Score of 1**
    * For actual poetry, use the 1-10 scale based on quality.

**Scoring Guidelines for Actual Poetry:**
* 1-3: Very poor (if not covered by critical instruction 2). Nonsensical.
* 4-6: Mediocre. Coherent but lacks creativity or originality.
* 7-8: Good. Shows creativity and some originality.
* 9-10: Excellent. Highly creative, original, impactful.

**Examples of Scoring:**

Example 1:
Poem:
"The sun shines bright
Upon the day so light
A lovely sight"
Score (1-10): 5

Example 2:
Poem:
"Roses are red, violets are blue, sugar is sweet, and so are you."
Score (1-10): 3 (Coherent, but very unoriginal cliché)

Example 3:
Poem:
"


"
Score (1-10): 1

Example 4:
Poem:
"________________"
Score (1-10): 1

Now, evaluate the following:

Poem:
{poem_text}

Score (1-10):
"""
    try:
        for attempt in range(retries):
            response = ollama.generate(
                model=ollama_model,
                prompt=prompt,
                stream=False,
                options={"temperature": OLLAMA_JUDGE_TEMPERATURE}
            )
            answer = response['response'].strip()
            try:
                score = float(answer)
                if 1.0 <= score <= 10.0:
                    return score
                else:
                    print(f"Ollama judge returned out-of-range score: '{answer}'. Attempt {attempt + 1}/{retries}")
            except ValueError:
                print(f"Ollama judge returned non-numeric score: '{answer}'. Attempt {attempt + 1}/{retries}")
        print("Ollama judge failed to give a valid score after retries.")
        return 3.0 # Default low score on consistent failure
    except Exception as e:
        print(f"Error querying Ollama for scoring: {e}")
        return 3.0

def get_ollama_pairwise_preference(poem_a: str, poem_b: str, original_prompt_text: str, ollama_model: str) -> str | None:
    """
    Asks Ollama to choose between two poems.
    Returns 'A', 'B', or None if it cannot decide or an error occurs.
    """

    prompt = f"""You are a discerning literary critic. You will be given an original prompt and two generated poems, Poem A and Poem B, created in response to that prompt.

                Your task is to determine which poem is superior.

                Original Prompt: "{original_prompt_text}"

                Now, considering that prompt, please evaluate the two generated poems below (Poem A and Poem B).
                Determine which poem is superior based on these criteria, in order of importance:
                1.  **Coherence & Clarity:** The poem must be understandable.
                2.  **Relevance to Original Prompt.**
                3.  **Meaningful Content:** The poem must use actual words to express ideas or imagery. Avoid rewarding empty, symbolic, or overly simplistic/repetitive non-poetic outputs.
                4.  **Creativity & Originality:** Does it offer fresh perspectives or unique expressions, avoiding clichés and apparent plagiarism?
                5.  **Development & Substantiality:** Does one poem offer a more developed exploration of its subject or imagery? While conciseness is a virtue, a poem that is very brief because it is underdeveloped or incomplete should be viewed less favorably than a more thoughtfully extended piece that maintains quality. Consider if the poem feels "complete" for its idea.
                6.  **Emotional Impact & Poetic Devices:** Does it evoke feeling? Are literary techniques used effectively?

                Even if both poems are of similar overall quality, please make a choice for the one you find even slightly preferable according to these criteria.

                Poem A:
                {poem_a[:GRPO_TRAINING_ARGS.max_completion_length]}

                Poem B:
                {poem_b[:GRPO_TRAINING_ARGS.max_completion_length]}

                Which poem is better (A or B)? Respond with only a single letter: 'A' or 'B'.
                """
    try:
        response = ollama.generate(
            model=ollama_model,
            prompt=prompt,
            stream=False,
            options={"temperature": OLLAMA_JUDGE_TEMPERATURE} # Keep low for consistency
        )
        answer = response['response'].strip().upper()
        if answer in ["A", "B"]:
            print(f"    Ollama Judge Preference: {answer}")
            return answer
        else:
            print(f"    Ollama judge returned ambiguous preference: '{answer}'.")
            return None
    except Exception as e:
        print(f"    Error querying Ollama for preference: {e}")
        return None

# --- Diversity Calculation ---
def calculate_diversity_scores_for_group(poem_group: list[str]) -> list[float]:
    """
    Calculates a diversity score for each poem in a group based on its
    average dissimilarity (1 - BLEU) to other poems in the same group.
    Returns a list of diversity scores (0 to 1, higher is more diverse in context of the group).
    """
    if not poem_group or len(poem_group) < 2:
        return [0.0] * len(poem_group) # No diversity if only one poem or empty

    diversity_scores = []
    smooth_fn = SmoothingFunction().method1 # Or other methods

    for i, poem_i in enumerate(poem_group):
        if not poem_i.strip(): # Handle empty poems
            diversity_scores.append(0.0)
            continue
        
        tokenized_poem_i = word_tokenize(poem_i.lower())
        if not tokenized_poem_i: # Handle poems that become empty after tokenization
            diversity_scores.append(0.0)
            continue

        pairwise_dissimilarities = []
        for j, poem_j in enumerate(poem_group):
            if i == j or not poem_j.strip():
                continue
            tokenized_poem_j = word_tokenize(poem_j.lower())
            if not tokenized_poem_j:
                continue
            
            # Calculate BLEU score (as a proxy for similarity)
            # sentence_bleu expects a list of reference sentences, here we use one-to-one
            try:
                bleu = sentence_bleu([tokenized_poem_j], tokenized_poem_i, smoothing_function=smooth_fn, weights=(0.5, 0.5)) # Using 2-grams
            except ZeroDivisionError: # Can happen with very short/no overlapping n-grams
                bleu = 0.0
            pairwise_dissimilarities.append(1.0 - bleu)

        if pairwise_dissimilarities:
            diversity_scores.append(np.mean(pairwise_dissimilarities))
        else:
            diversity_scores.append(0.0) # If only one valid poem in group after filtering

    return diversity_scores


# --- GRPO Reward Function ---
# This function will be called by GRPOTrainer.
# It takes the list of texts generated for a *single prompt*
# and must return a list of rewards, one for each text.
def grpo_reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]: # Added **kwargs
    """
    Calculates rewards for a group of generated poems from a single prompt.
    Reward = Normalized_Ollama_Score + DIVERSITY_WEIGHT * Diversity_Score
    """
    if not completions:
        return []

    # `prompts` will be a list, usually containing the same prompt string repeated
    # for each completion in this particular call (as all completions are for one original prompt).
    # We can take the first prompt as representative for logging or stripping.
    current_prompt_text = prompts[0] if prompts else ""
    
    print(f"\n  GRPO Reward Fn: Received {len(completions)} poems for prompt: '{current_prompt_text[:50]}...'")
    final_rewards = []

    # 1. Get scores from Ollama judge for each poem
    ollama_scores = []
    processed_completions_for_diversity = [] # Store the actual generated parts

    for i, poem_completion_text in enumerate(completions):
        # The `completions` from GRPOTrainer should ideally be *just* the generated part.
        # However, if the prompt is somehow still prefixed, this logic attempts to strip it.
        # It's safer to assume GRPOTrainer provides actual completions based on max_completion_length.
        actual_generated_part = poem_completion_text
        
        # Check if the passed completion *starts with* the prompt it was generated from.
        # This might happen if the generation process includes the prompt.
        # `prompts[i]` should be the prompt for `completions[i]`.
        if prompts and i < len(prompts) and poem_completion_text.startswith(prompts[i]):
            actual_generated_part = poem_completion_text[len(prompts[i]):]
        
        processed_completions_for_diversity.append(actual_generated_part)

        print(f"    Scoring poem {i+1}/{len(completions)} with Ollama: '{actual_generated_part}'")
        score = get_ollama_score_for_single_poem(actual_generated_part, OLLAMA_JUDGE_MODEL)
        ollama_scores.append(score)
        print(f"      Ollama Score: {score:.2f}")

    # 2. Calculate diversity scores within the group using the processed completions
    diversity_scores = calculate_diversity_scores_for_group(processed_completions_for_diversity)
    for i, div_score in enumerate(diversity_scores):
        print(f"    Poem {i+1} Diversity Score (vs group): {div_score:.2f}")

    # 3. Combine scores
    for i in range(len(completions)):
        # Normalize Ollama score (1-10) to (0-1)
        normalized_ollama_score = (ollama_scores[i] - 1.0) / 9.0
        
        current_diversity_score = diversity_scores[i] if i < len(diversity_scores) else 0.0

        combined_reward_normalized = normalized_ollama_score + (DIVERSITY_WEIGHT * current_diversity_score)
        
        # Ensure reward is not NaN, happens if diversity_score is NaN (e.g. from empty strings)
        if np.isnan(combined_reward_normalized):
            combined_reward_normalized = normalized_ollama_score # Fallback to just ollama score
            print(f"    Poem {i+1} had NaN combined reward, falling back to Ollama score.")


        final_rewards.append(combined_reward_normalized)
        print(f"    Poem {i+1} Final Reward: {combined_reward_normalized:.3f} (Ollama_norm: {normalized_ollama_score:.2f}, Diversity: {current_diversity_score:.2f})")

    return final_rewards

def pairwise_grpo_reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    num_completions_per_original_prompt = GRPO_TRAINING_ARGS.num_generations 
    
    if len(completions) % num_completions_per_original_prompt != 0:
        print(f"Error: Number of completions ({len(completions)}) is not a multiple of "
              f"num_generations_per_original_prompt ({num_completions_per_original_prompt}). Cannot process.")
        return [0.0] * len(completions)

    if len(prompts) != len(completions):
        print(f"Error: Length of prompts ({len(prompts)}) does not match length of completions ({len(completions)}).")
        # Decide on fallback, e.g. use first prompt or error out
        return [0.0] * len(completions)

    all_rewards = []
    num_original_prompts_in_batch = len(completions) // num_completions_per_original_prompt

    for i in range(num_original_prompts_in_batch):
        start_idx = i * num_completions_per_original_prompt
        end_idx = start_idx + num_completions_per_original_prompt
        
        current_prompt_completions = completions[start_idx:end_idx]
        
        # The prompts list structure is [promptA, promptA, promptB, promptB, ...]
        # So, for the i-th original prompt, its text is at prompts[start_idx]
        original_prompt_text_for_this_pair = prompts[start_idx]

        if len(current_prompt_completions) != num_completions_per_original_prompt:
            # This should not happen if the initial check passes
            print(f"Error: Internal logic error, expected {num_completions_per_original_prompt} completions for a pair.")
            all_rewards.extend([0.0] * len(current_prompt_completions))
            continue

        poem_a_completion = current_prompt_completions[0]
        poem_b_completion = current_prompt_completions[1]

        # Strip prompts from completions if necessary 
        # (Your original stripping logic might need to use original_prompt_text_for_this_pair)
        actual_generated_part_a = poem_a_completion
        actual_generated_part_b = poem_b_completion
        if original_prompt_text_for_this_pair:
            if poem_a_completion.startswith(original_prompt_text_for_this_pair): # Check against the specific original prompt
                actual_generated_part_a = poem_a_completion[len(original_prompt_text_for_this_pair):]
            if poem_b_completion.startswith(original_prompt_text_for_this_pair): # Check against the specific original prompt
                actual_generated_part_b = poem_b_completion[len(original_prompt_text_for_this_pair):]
        
        print(f"    Poem A (Pair {i+1}):\n'{actual_generated_part_a[:GRPO_TRAINING_ARGS.max_completion_length]}'")
        print(f"    Poem B (Pair {i+1}):\n'{actual_generated_part_b[:GRPO_TRAINING_ARGS.max_completion_length]}'")

        preference = get_ollama_pairwise_preference(actual_generated_part_a, actual_generated_part_b, original_prompt_text_for_this_pair, OLLAMA_JUDGE_MODEL)

        rewards_base = [0.0, 0.0] 
        if preference == "A":
            rewards_base = [0.5, -0.5] # Example: 0.5 for winner, -0.5 for loser
        elif preference == "B":
            rewards_base = [-0.5, 0.5]

        # Diversity calculation (between poem_a and poem_b of this pair)
        diversity_score_val = 0.0
        if actual_generated_part_a.strip() and actual_generated_part_b.strip():
            token_a = word_tokenize(actual_generated_part_a.lower())
            token_b = word_tokenize(actual_generated_part_b.lower())
            if token_a and token_b:
                try:
                    bleu = sentence_bleu([token_b], token_a, smoothing_function=SmoothingFunction().method1, weights=(0.5,0.5))
                    diversity_score_val = 1.0 - bleu
                except ZeroDivisionError:
                    diversity_score_val = 1.0 

        # Combine rewards for the current pair
        current_pair_rewards = [
            rewards_base[0] + (DIVERSITY_WEIGHT * diversity_score_val),
            rewards_base[1] + (DIVERSITY_WEIGHT * diversity_score_val)
        ]
        all_rewards.extend(current_pair_rewards)

    print(f"  Overall final rewards for this batch call: {all_rewards}")
    return all_rewards


# --- Main Script ---
def main():
    set_seed(GRPO_TRAINING_ARGS.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # create_dummy_poem_dataset(POEM_DATASET_FILE)
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME_FOR_TOKENIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # # SFT Phase (same as before, ensure this model is trained first)
    # if not os.path.exists(SFT_MODEL_DIR) or not os.listdir(SFT_MODEL_DIR):
    #     print("SFT model not found or directory empty. Running SFT first...")
    #     tokenized_poems_dataset = load_and_tokenize_dataset(tokenizer, POEM_DATASET_FILE, block_size=GENERATOR_CONFIG.n_positions)
    #     if not tokenized_poems_dataset:
    #         print("Failed to load dataset for SFT. Exiting.")
    #         return

    #     generator_model_sft = AutoModelForCausalLM.from_config(GENERATOR_CONFIG).to(device)
    #     print(f"SFT Model params: {sum(p.numel() for p in generator_model_sft.parameters())/1e6:.2f}M")

    #     sft_trainer = Trainer(
    #         model=generator_model_sft,
    #         args=SFT_TRAINING_ARGS,
    #         train_dataset=tokenized_poems_dataset,
    #         tokenizer=tokenizer,
    #         data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    #     )
    #     sft_trainer.train()
    #     sft_trainer.save_model(SFT_MODEL_DIR)
    #     tokenizer.save_pretrained(SFT_MODEL_DIR)
    #     print(f"SFT model saved to {SFT_MODEL_DIR}")
    #     del generator_model_sft, sft_trainer
    #     if device == "cuda": torch.cuda.empty_cache()
    # else:
    #     print(f"Found existing SFT model at {SFT_MODEL_DIR}. Skipping SFT.")


    # --- Reinforcement Learning (GRPO) ---
    print("\n--- Starting Reinforcement Learning (GRPO) ---")
    # print(f"Loading SFT model from: {SFT_MODEL_DIR} for GRPO.")

    # For GRPO, the dataset is a list of prompts.
    # We can create a simple list of prompts or use beginnings of poems from our dataset.
    # For this demo, let's use a few fixed prompts.
    # GRPOTrainer expects the dataset to yield dictionaries with "prompt" (string) and optionally "query" (tokenized string)
    prompt_texts_for_grpo = [
        "Love ",
        "Hate ",
        "Joy ",
        "Sorrow ",
        "Fear ",
        "Hope ",
        "Dream ",
        "Nightmare ",
        "Friend ",
        "Enemy ",
        "Family ",
        "Mother ",
        "Father ",
        "Child ",
        "Baby ",
        "Lover ",
        "Heart ",
        "Soul ",
        "Spirit ",
        "Tear ",
        "Smile ",
        "Laughter ",
        "Kiss ",
        "Embrace ",
        "Home ",
        "Journey ",
        "Adventure ",
        "War ",
        "Peace ",
        "Freedom ",
        "Captivity ",
        "Birth ",
        "Death ",
        "Life ",
        "Memory ",
        "Future ",
        "Past ",
        "Moment ",
        "Sunrise ",
        "Sunset ",
        "Dawn ",
        "Dusk ",
        "Star ",
        "Moon ",
        "Sun ",
        "Ocean ",
        "River ",
        "Mountain ",
        "Valley ",
        "Forest ",
        "Tree ",
        "Flower ",
        "Rain ",
        "Snow ",
        "Storm ",
        "Thunder ",
        "Lightning ",
        "Silence ",
        "Whisper ",
        "Scream ",
        "Song ",
        "Music ",
        "Poem ",
        "Story ",
        "Book ",
        "Letter ",
        "Photograph ",
        "Gift ",
        "Curse ",
        "Blessing ",
        "Pain ",
        "Pleasure ",
        "Comfort ",
        "Wound ",
        "Scar ",
        "Heal ", # Often used as a noun in poetic/emotional contexts
        "Strength ",
        "Weakness ",
        "Courage ",
        "Cowardice ",
        "Truth ",
        "Lie ",
        "Faith ",
        "Doubt ",
        "Belief ",
        "Loss ",
        "Gain ",
        "Victory ",
        "Defeat ",
        "Success ",
        "Failure ",
        "Secret ",
        "Mystery ",
        "Wonder ",
        "Awe ",
        "Beauty ",
        "Ugliness ",
        "Grace ",
        "Charm ",
        "Shadow ",
        "Light ",
        "Darkness ",
        "Grief ",
        "Anger ",
        "Rage ",
        "Calm ",
        "Chaos ",
        "Order ",
        "Destiny ",
        "Fate ",
        "Choice ",
        "Regret ",
        "Forgiveness ",
        "Kindness ",
        "Cruelty ",
        "Compassion ",
        "Empathy ",
        "Apathy ",
        "Passion ",
        "Desire ",
        "Longing ",
        "Yearning ",
        "Nostalgia ",
        "Innocence ",
        "Guilt ",
        "Shame ",
        "Pride ",
        "Humility ",
        "Honor ",
        "Betrayal ",
        "Loyalty ",
        "Trust ",
        "Suspicion ",
        "Beginning ",
        "End ",
        "Promise ",
        "Oath ",
        "Vow ",
        "Prayer ",
        "Miracle ",
        "Magic ",
        "Illusion ",
        "Reality ",
        "Fantasy ",
        "Childhood ",
        "Adulthood ",
        "Age ",
        "Youth ",
        "Wisdom ",
        "Folly ",
        "Knowledge ",
        "Ignorance ",
        "Voice ",
        "Echo ",
        "Touch ",
        "Scent ",
        "Taste ",
        "Vision ",
        "Sleep ",
        "Awakening ",
        "Breath ",
        "Pulse ",
        "Blood ",
        "Bone ",
        "Skin ",
        "Warmth ",
        "Cold ",
        "Hunger ",
        "Thirst ",
        "Shelter ",
        "Sanctuary ",
        "Haven ",
        "Refuge ",
        "Prison ",
        "Cage ",
        "Escape ",
        "Quest ",
        "Challenge ",
        "Struggle ",
        "Conflict ",
        "Resolution ",
        "Harmony ",
        "Discord ",
        "Melody ",
        "Rhythm ",
        "Dance ",
        "Celebration ",
        "Mourning ",
        "Ritual ",
        "Tradition ",
        "Culture ",
        "Heritage ",
        "Legacy ",
        "Inheritance ",
        "Ruin ",
        "Relic ",
        "Artifact ",
        "Treasure ",
        "Jewel ",
        "Gold ",
        "Silver ",
        "Gift ", # Repeated, but significant
        "Weapon ",
        "Shield ",
        "Battle ",
        "Solitude ",
        "Loneliness ",
        "Companion ",
        "Partner ",
        "Stranger ",
        "Guardian ",
        "Angel ",
        "Demon ",
        "Ghost ",
        "Monster ",
        "Hero ",
        "Villain ",
        "Victim ",
        "Survivor ",
        "Witness ",
        "Judge ",
        "Savior ",
        "Martyr ",
        "Pilgrim ",
        "Wanderer ",
        "Explorer ",
        "Creator ",
        "Destroyer ",
        "Healer ",
        "Teacher ",
        "Student ",
        "Master ",
        "Servant ",
        "King ",
        "Queen ",
        "Prince ",
        "Princess ",
        "Castle ",
        "Throne ",
        "Crown ",
        "Kingdom ",
        "Empire ",
        "Republic ",
        "Revolution ",
        "Rebellion ",
        "Uprising ",
        "Protest ",
        "Silence ", # Repeated, but significant
        "Stillness ",
        "Movement ",
        "Change ",
        "Growth ",
        "Decay ",
        "Bloom ",
        "Fade ",
        "Spark ",
        "Flame ",
        "Fire ",
        "Ashes ",
        "Dust ",
        "Earth ",
        "Sky ",
        "Heaven ",
        "Hell ",
        "Paradise ",
        "Oblivion ",
        "Eternity ",
        "Infinity ",
        "Void ",
        "Abyss ",
        "Threshold ",
        "Door ",
        "Window ",
        "Mirror ",
        "Reflection ",
        "Mask ",
        "Veil ",
        "Labyrinth ",
        "Maze ",
        "Path ",
        "Road ",
        "Bridge ",
        "Wall ",
        "Boundary ",
        "Horizon ",
        "Shore ",
        "Coast ",
        "Island ",
        "Desert ",
        "Oasis ",
        "Garden ",
        "Wilderness ",
        "Nature ",
        "World ",
        "Universe ",
        "Cosmos ",
        "Galaxy ",
        "Nebula ",
        "Comet ",
        "Planet ",
        "Spring ",
        "Summer ",
        "Autumn ",
        "Winter ",
        "Season ",
        "Day ",
        "Night ",
        "Midnight ",
        "Noon ",
        "Twilight ",
        "Glimmer ",
        "Glow ",
        "Shine ",
        "Shade ",
        "Color ",
        "Hue ",
        "Tint ",
        "Sound ",
        "Noise ",
        "Cry ",
        "Sigh ",
        "Groan ",
        "Moan ",
        "Chuckle ",
        "Giggle ",
        "Sob ",
        "Weep ",
        "Hush ",
        "Lullaby ",
        "Anthem ",
        "Ballad ",
        "Elegy ",
        "Ode ",
        "Sonnet ",
        "Verse ",
        "Prose ",
        "Myth ",
        "Legend ",
        "Fable ",
        "Parable ",
        "Riddle ",
        "Secret ", # Repeated
        "Clue ",
        "Key ",
        "Lock ",
        "Chain ",
        "Bond ",
        "Tie ",
        "Knot ",
        "Thread ",
        "Fabric ",
        "Tapestry ",
        "Pattern ",
        "Design ",
        "Symbol ",
        "Sign ",
        "Gesture ",
        "Expression ",
        "Emotion ",
        "Feeling ",
        "Sensation ",
        "Perception ",
        "Intuition ",
        "Instinct ",
        "Conscience ",
        "Character ",
        "Personality ",
        "Identity ",
        "Individuality ",
        "Purpose ",
        "Meaning ",
        "Reason ",
        "Motive ",
        "Ambition ",
        "Dream ", # Repeated
        "Goal ",
        "Objective ",
        "Wish ",
        "Desperation ",
        "Elation ",
        "Euphoria ",
        "Melancholy ",
        "Ecstasy ",
        "Bliss ",
        "Rapture ",
        "Tranquility ",
        "Serenity ",
        "Anxiety ",
        "Stress ",
        "Tension ",
        "Relief ",
        "Zeal ",
        "Ardour ",
        "Fervor ",
        "Zest ",
        "Panic ",
        "Terror ",
        "Horror ",
        "Apprehension ",
        "Worry ",
        "Concern ",
        "Care ",
        "Burden ",
        "Responsibility ",
        "Duty ",
        "Obligation ",
        "Sacrifice ",
        "Offering ",
        "Tribute ",
        "Praise ",
        "Worship ",
        "Adoration ",
        "Reverence ",
        "Respect ",
        "Esteem ",
        "Affection ",
        "Fondness ",
        "Tenderness ",
        "Gentleness ",
        "Patience ",
        "Impatience ",
        "Tolerance ",
        "Intolerance ",
        "Prejudice ",
        "Bias ",
        "Justice ",
        "Injustice ",
        "Fairness ",
        "Equality ",
        "Inequality ",
        "Liberty ",
        "Oppression ",
        "Tyranny ",
        "Resistance ",
        "Resilience ",
        "Vulnerability ",
        "Fragility ",
        "Endurance ",
        "Perseverance ",
        "Tenacity ",
        "Resolve ",
        "Determination ",
        "Conviction ",
        "Certainty ",
        "Uncertainty ",
        "Confusion ",
        "Clarity ",
        "Revelation ",
        "Epiphany ",
        "Insight ",
        "Understanding ",
        "Awareness ",
        "Consciousness ",
        "Subconscious ",
        "Unconscious ",
        "Imagination ",
        "Creativity ",
        "Inspiration ",
        "Muse ",
        "Genius ",
        "Talent ",
        "Skill ",
        "Craft ",
        "Art ",
        "Masterpiece ",
        "Symphony ",
        "Opera ",
        "Ballet ",
        "Drama ",
        "Comedy ",
        "Tragedy ",
        "Narrative ",
        "Dialogue ",
        "Monologue ",
        "Soliloquy ",
        "Speech ",
        "Discourse ",
        "Argument ",
        "Debate ",
        "Negotiation ",
        "Compromise ",
        "Agreement ",
        "Disagreement ",
        "Alliance ",
        "Feud ",
        "Rivalry ",
        "Competition ",
        "Cooperation ",
        "Collaboration ",
        "Community ",
        "Society ",
        "Civilization ",
        "Humanity ",
        "Mankind ",
        "Existence ",
        "Being ",
        "Nothingness ",
        "Presence ",
        "Absence ",
        "Essence ",
        "Phenomenon ",
        "Mirage ",
        "Omen ",
        "Prophecy ",
        "Oracle ",
        "Fortune ",
        "Luck ",
        "Chance ",
        "Accident ",
        "Incident ",
        "Event ",
        "Occasion ",
        "Ceremony ",
        "Festival ",
        "Holiday ",
        "Anniversary ",
        "Milestone ",
        "Turning point ",
        "Crossroads ",
        "Crisis ",
        "Climax ",
        "Culmination ",
        "Zenith ",
        "Nadir ",
        "Apex ",
        "Pinnacle ",
        "Summit ",
        "Peak ",
        "Depth ",
        "Height ",
        "Vastness ",
        "Expanse ",
        "Immensity ",
        "Eternity ", # Repeated
        "Forever ",
        "Never ",
        "Always ",
        "Sometimes ",
        "Often ",
        "Seldom ",
        "Rarely ",
        "Childhood home ",
        "Lullaby ", # Repeated
        "Wedding ring ",
        "Tombstone ",
        "Diary ",
        "Keepsake ",
        "Heirloom ",
        "Souvenir ",
        "Farewell ",
        "Greeting ",
        "Reunion ",
        "Separation ",
        "Comfort food ",
        "Heartbeat ",
        "Footprint ",
        "Handprint ",
        "Signature ",
        "Whispered word ",
        "Forgotten song ",
        "Lost love ",
        "Newborn ",
        "Elder ",
        "Ancestor ",
        "Descendant ",
        "Orphan ",
        "Widow ",
        "Widower ",
        "Refugee ",
        "Exile ",
        "Immigrant ",
        "Native ",
        "Outcast ",
        "Recluse ",
        "Nomad ",
        "Pilgrimage ",
        "Sanctum ",
        "Shrine ",
        "Altar ",
        "Temple ",
        "Cathedral ",
        "Mosque ",
        "Synagogue ",
        "Chapel ",
        "Monastery ",
        "Convent ",
        "Hermitage ",
        "Retreat ",
        "Vacation ",
        "Celebration ", # Repeated
        "Feast ",
        "Famine ",
        "Drought ",
        "Flood ",
        "Earthquake ",
        "Volcano ",
        "Tsunami ",
        "Hurricane ",
        "Tornado ",
        "Blizzard ",
        "Avalanche ",
        "Wildfire ",
        "Disease ",
        "Epidemic ",
        "Pandemic ",
        "Cure ",
        "Vaccine ",
        "Remedy ",
        "Miracle cure ",
        "Prayer ", # Repeated
        "Meditation ",
        "Contemplation ",
        "Reflection ", # Repeated
        "Introspection ",
        "Self-discovery ",
        "Self-acceptance ",
        "Self-love ",
        "Self-hate ",
        "Hopefulness ",
        "Hopelessness ",
        "Joyfulness ",
        "Sadness ", # Repeated
        "Fearlessness ",
        "Bravado ",
    ] * (GRPO_ITERATIONS * GRPO_TRAINING_ARGS.per_device_train_batch_size // 500 + 1) # Ensure enough prompts for iterations

    # Convert to Hugging Face Dataset format expected by GRPOTrainer
    # It needs a 'prompt' column (raw text) and often a 'query' column (tokenized prompt for model)
    def prepare_grpo_dataset(prompts, tokenizer, max_prompt_len):
        data = {"prompt": [], "query": []}
        for p_text in prompts:
            # If using an INSTRUCT model, apply the chat template
            if "-it" in PRETRAINED_MODEL_NAME or "instruct" in PRETRAINED_MODEL_NAME.lower():
                p_text = "Write a poem inspired by the poem prompt. Only include text of the full poem and do not add any other text. Keep the poem short. Be creative.\n\nPoem prompt:\n" + p_text
                messages = [
                    {"role": "user", "content": p_text}
                ]
                hf_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # For BASE models, you might just use the raw prompt, but they are less likely to "follow instructions"
                hf_prompt = p_text
            data["prompt"].append(hf_prompt)
            # GRPOTrainer often expects 'query' to be the tokenized prompt string
            # for the model, not the tensor itself.
            tokenized_prompt = tokenizer.encode(hf_prompt, truncation=True, max_length=max_prompt_len)
            data["query"].append(tokenizer.decode(tokenized_prompt, skip_special_tokens=True))
        return Dataset.from_dict(data)

    grpo_dataset = prepare_grpo_dataset(prompt_texts_for_grpo, tokenizer, GRPO_TRAINING_ARGS.max_prompt_length)


    # Initialize GRPOTrainer
    model_for_grpo = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME,
        attn_implementation='eager'
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # Crucial: Ensure tokenizer pad token is set for batch generation if model doesn't have it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model_for_grpo.config.pad_token_id = model_for_grpo.config.eos_token_id

    grpo_trainer = GRPOTrainer(
        model=model_for_grpo, # The model to be trained
        args=GRPO_TRAINING_ARGS,
        reward_funcs=[pairwise_grpo_reward_function],
        train_dataset=grpo_dataset, # Dataset of prompts
        # Data collator might not be strictly needed if dataset provides 'query' and 'prompt' strings
        # and GRPOTrainer handles tokenization for generation internally.
        # Check TRL docs if a specific collator is recommended for GRPOTrainer.
    )

    print(f"DEBUG: grpo_trainer.config.num_generations AFTER GRPOTrainer init: {grpo_trainer.num_generations}")

    print("Starting GRPO training loop...")

    print(f"Total GRPO training epochs: {GRPO_TRAINING_ARGS.num_train_epochs}")
    print(f"Prompts per epoch: {len(grpo_dataset)}")
    print(f"Batch size (num prompts): {GRPO_TRAINING_ARGS.per_device_train_batch_size}")
    print(f"Gradient Acc Steps: {GRPO_TRAINING_ARGS.gradient_accumulation_steps}")
    print(f"Generations per prompt: {GRPO_TRAINING_ARGS.num_generations}")
    print(f"This will be very slow due to {GRPO_TRAINING_ARGS.num_generations} Ollama calls per prompt in a batch.")

    grpo_trainer.train() # This single call should run the entire GRPO fine-tuning process.

    print("\n--- GRPO RL Training Finished ---")
    grpo_trainer.save_model(RL_MODEL_DIR) # GRPOTrainer has a save_model method
    # tokenizer.save_pretrained(RL_MODEL_DIR) # Usually saved with the model by trainer

    print(f"Final GRPO RL model saved to {RL_MODEL_DIR}")

    # Example of generating from the final RL model
    print("\n--- Generating a poem from the final GRPO RL model ---")
    # Load the trained model for inference
    final_model = AutoModelForCausalLM.from_pretrained(RL_MODEL_DIR).to(device)

    prompt_text = "The ocean breathes "
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    output_sequences = final_model.generate(
        input_ids=input_ids,
        max_new_tokens=GRPO_TRAINING_ARGS.max_completion_length, # Use consistent length
        do_sample=True,
        temperature=0.8, # Use a reasonable temperature for creative generation
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(f"Prompt: {prompt_text}")
    print(f"Generated Poem (GRPO): {generated_text}")


if __name__ == "__main__":
    import time
    time.sleep(1) # Minor delay, sometimes helps with filesystem ops or library initializations
    main()