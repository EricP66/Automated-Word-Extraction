import docx
import pickle

def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# Usage
file_path = 'Biased_text1.docx'
text = read_docx(file_path)
# print(text)

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize the text into sentences or words
sentences = text.split('.')

# Example function to check if the sentence contains any of the WEAT words
def find_sentences_with_words(sentences, word_list):
    selected_sentences = []
    for sentence in sentences:
        for word in word_list:
            if word in sentence:
                selected_sentences.append(sentence)
                break
    return selected_sentences

# Word Sets for text 1
target1 = ["Chinese", "Japan", "Eastern countries"] 
target2 = ["American", "United States", "Western countries", "Canada"] 
target3 = ["Indian"] 
target4 = ["African"] 
target5 = ["European"]

attribute1 = ["respect for authority", "structured", "diligence", "academic rigor", "systematic study", "precision", "strong work ethic"]
attribute2 = ["interactive", "open atmosphere", "independence", "critical thinking", "ownership", "assertive", "facilitative"] 
attribute3 = ["mathematics", "engineering", "high proficiency", "societal expectations"]
attribute4 = ["structured", "respectful", "discipline", "attentiveness", "collective spirit", "cohesive"] 
attribute5 = ["individual personalities", "freedom", "lively", "varied", "individual achievement"]

# # Word Sets for text 2
# target1 = ["Mexico", "Carlos Ramirez"]
# target2 = ["United States", "Sarah Thompson"]
# target3 = ["China", "Li Wei"]
# target4 = ["Nigerian", "Aisha Mohammed"]
# target5 = ["Germany", "Michael Jensen"]
# target6 = ["Indian-American", "Priya Patel"]

# attribute1 = ["talent", "discipline", "perseverance", "analytical mind", "natural ability"]
# attribute2 = ["creativity", "individualism", "social consciousness"]
# attribute3 = ["discipline", "academic achievement", "precision", "commitment", "long-term goals"]
# attribute4 = ["creativity", "resilience", "innovation"]
# attribute5 = ["hands-on", "practical skills", "technical expertise", "applied learning", "vocational training"]
# attribute6 = ["innovation", "cultural heritage", "modern medical practices", "holistic approaches"]

# # Word Sets for text 3
# target1 = ["American", "United States"]
# target2 = ["East Asian", "South Korea", "Japan", "China"]
# target3 = ["European", "Germany", "Switzerland"]
# target4 = ["African", "Nigeria", "Kenya"]
# target5 = ["Indian"]

# attribute1 = ["independently", "innovate", "entrepreneurial", "creativity", "pioneering", "independence", "critical thinking", "innovation", "entrepreneurship", "individuality", "risk-taking"]
# attribute2 = ["diligence", "technical precision", "rigorous", "competitive", "precision", "technical expertise", "disciplined", "structured", "perseverance", "meticulous"]
# attribute3 = ["applied sciences", "hands-on", "technical knowledge", "practical experience", "problem-solving", "practical approach"]
# attribute4 = ["unique perspective", "cultural values", "sustainable development", "resilience", "innovate", "limited resources"]
# attribute5 = ["quantitative skills", "mathematical rigor", "analytical skills", "technical proficiency"]

# # Word Sets for text 4
# target1 = ["American", "United States"]
# target2 = ["East Asian", "South Korea", "Japan", "China"]
# target3 = ["European", "France", "Germany", "Italy", "Switzerland"]
# target4 = ["African", "Nigeria", "Kenya"]
# target5 = ["Indian", "India"]

# attribute1 = ["competitive", "leadership", "critical thinking", "public speaking", "teamwork", "resilience"]
# attribute2 = ["academic-focused", "achievement", "personal discipline", "dedication", "rigorous", "study habits", "academic success", "technical skills"]
# attribute3 = ["creativity", "cultural expression", "artistic appreciation", "personal expression", "cultural themes", "creativity", "individuality"]
# attribute4 = ["athletic", "physical fitness", "community", "teamwork", "resilience", "pride", "disciplined", "dedicated"]
# attribute5 = ["social service", "community involvement", "social welfare", "social awareness", "responsibility", "commitment", "social impact", "collective welfare"]

# # Ground truth 1
# target1 = ["Eastern", "China", "Japan", "Chinese"]
# target2 = ["Western", "United States", "Canada", "American"]
# target3 = ["Indian"]
# target4 = ["African"]
# target5 = ["European"]

# attribute1 = ["respect for authority", "structured approach", "attentively", "detailed notes", "diligence", "academic rigor", "strong work ethic", "precision"]
# attribute2 = ["interactive", "open", "independence", "critical thinking", "ownership", "assertively", "leadership", "confidence"]
# attribute3 = ["societal expectations", "mathematics", "engineering", "proficiency"]
# attribute4 = ["structured", "respectful environment", "discipline", "attentiveness", "collective spirit", "cohesive"]
# attribute5 = ["individual personalities", "freedom", "lively", "varied", "individual achievement"]


# # Ground truth 2 
# target1 = ["Carlos Ramirez", "Mexico"]
# target2 = ["Sarah Thompson", "United States"]
# target3 = ["Li Wei", "China"]
# target4 = ["Aisha Mohammed", "Nigerian"]
# target5 = ["Michael Jensen", "Germany"]
# target6 = ["Priya Patel", "Indian-American"]

# attribute1 = ["discipline", "perseverance", "competitive", "analytical mind", "talent", "supportive", "structured"]
# attribute2 = ["civic engagement", "empathy", "leadership", "eloquence", "individualism", "social consciousness"]
# attribute3 = ["Precision", "Perseverance", "dedication", "methodical approach", "discipline", "academic achievement", "commitment"]
# attribute4 = ["limited resources", "creativity", "resilience", "inspiring", "break traditional gender norms"]
# attribute5 = ["hands-on approach", "practical skills", "technical expertise", "applied learning", "vocational training"]
# attribute6 = ["Innovation", "holistic approaches", "cultural heritage", "modern ambition"]

# # Ground truth 3
# target1 = ["American", "United States"]
# target2 = ["East Asian", "South Korea", "Japan", "China"]
# target3 = ["European", "Germany", "Switzerland"]
# target4 = ["African", "Nigeria", "Kenya"]
# target5 = ["Indian"]

# attribute1 = ["think independently", "innovate", "entrepreneurial", "creativity", "pioneering ideas", "critical thinking", "innovation", "entrepreneurship", "individuality", "risk-taking"]
# attribute2 = ["Precision", "Diligence", "academic success", "rigorous", "focused", "competitive", "expertise", "disciplined", "meticulous", "perseverance", "structured"]
# attribute3 = ["applied sciences", "Practical", "hands-on skills", "technical knowledge", "apprenticeship", "real-world settings", "problem-solving abilities"]
# attribute4 = ["cultural values", "sustainable development", "cultural emphasis", "societal benefit", "resilience", "limited resources"]
# attribute5 = ["Mathematical Excellence", "quantitative skills", "mathematical rigor", "analytical skills", "technical proficiency"]

# # Ground truth 4
# target1 = ["American", "United States"]
# target2 = ["East Asian", "China", "South Korea", "Japan"]
# target3 = ["European", "France", "Germany", "Italy", "Switzerland"]
# target4 = ["African", "Nigeria", "Kenya"]
# target5 = ["Indian", "India"]

# attribute1 = ["competitive spirit", "leadership", "critical thinking", "public speaking", "teamwork", "resilience"]
# attribute2 = ["Academic Excellence", "achievement", "personal discipline", "dedication", "rigorous", "academic success", "technical skills", "collective commitment"]
# attribute3 = ["creativity", "cultural expression", "artistic appreciation", "personal expression", "cultural themes", "individuality"]
# attribute4 = ["Athletic Excellence", "Community Spirit", "physical fitness", "athletic achievements", "teamwork", "resilience", "pride", "disciplined", "dedicated"]
# attribute5 = ["social service", "community involvement", "cultural values", "social awareness", "responsibility", "collective welfare"]


target_words = target1 + target2 + target3 + target4 + target5 + attribute1 + attribute2 + attribute3 + attribute4 + attribute5 
target_sentences = find_sentences_with_words(sentences, target_words)

import torch
from transformers import GPT2LMHeadModel

model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model_gpt2.eval()
model_gpt2.to('cuda')


def gpt2_custom(sentences, word_list, out_name):
    # Load the pretrained GPT-2 model and tokenizer
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    model_gpt2.eval()
    model_gpt2.to('cuda')

    # Create a dictionary to store embeddings
    out_dict = {word: [] for word in word_list}
    # print(out_dict)
    for word in word_list:
        for sentence in sentences:
            if word in sentence:
                input_ids = torch.tensor(tokenizer_gpt2.encode(sentence, add_prefix_space=True)).unsqueeze(0).to('cuda')
                
                # Get the subword tokens for the target word
                word_tokens = tokenizer_gpt2.encode(word, add_prefix_space=True)

                # print(f"Sentence tokens for sentence '{sentence}': {input_ids[0].tolist()}")
                # print(f"Word tokens for '{word}': {word_tokens}")

                # Find the positions of the subword tokens in the input_ids
                positions = [i for i, token_id in enumerate(input_ids[0].tolist()) if token_id in word_tokens]

                if positions:
                    outputs = model_gpt2(input_ids)
                    hidden_states = outputs.hidden_states[-1]  # Last layer embeddings

                    # Get the average embeddings of all subword token embeddings
                    embeddings = []
                    for pos in positions:
                        embeddings.append(hidden_states[0, pos, :].cpu().detach().numpy())

                    avg_embedding = sum(embeddings)/len(embeddings)

                    # # Check if the embedding was successfully generated
                    # print(f"Generated embedding for word '{word}': {word_embedding[:5]}...")  # Print first few values

                    out_dict[word].append(avg_embedding)
                    print(f"Embeddings for word '{word}' so far: {len(out_dict[word])}")

                else:
                    print(f"Warning: No positions found for word '{word}' in sentence '{sentence}'")
            
    for word, embeddings in out_dict.items():
        if not embeddings:
            print(f"Warning: Embeddings for word '{word}' are empty.")

    # Save embeddings
    pickle.dump(out_dict, open(f'gpt2_{out_name}.pickle', 'wb'))

word_embeddings = gpt2_custom(target_sentences, target_words, 'text_case_1_auto')

