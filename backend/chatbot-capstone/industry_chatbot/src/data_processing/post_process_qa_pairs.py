import json
import re

def clean_augmented_dataset(input_file=r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\augmented_qa_pairs.json', 
                             output_file=r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\cleaned_augmented_qa_pairs.json'):
    """Clean and deduplicate augmented QA dataset"""
    with open(input_file, 'r', encoding='utf-8') as f:
        augmented_data = json.load(f)
    
    cleaned_data = []
    seen_questions = set()
    
    for entry in augmented_data:
        # Clean question
        question = re.sub(r'^(Rephrase this question:|\s+)', '', entry['question']).strip()
        question = re.sub(r'[.nrs]+$', '', question).strip()
        
        # Skip empty or duplicate questions
        if question and question not in seen_questions:
            entry['question'] = question
            cleaned_data.append(entry)
            seen_questions.add(question)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned dataset: {len(augmented_data)} → {len(cleaned_data)} entries")

if __name__ == "__main__":
    clean_augmented_dataset()