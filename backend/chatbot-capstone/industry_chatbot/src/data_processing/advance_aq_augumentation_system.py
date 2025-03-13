import json
import logging
from typing import List, Dict
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

class QADatasetEnhancer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            raise
    
    def generate_alternative_question(self, original_question: str) -> str:
        """Generate a single alternative question."""
        input_ids = self.tokenizer.encode(
            f"Rephrase this question: '{original_question}'", 
            return_tensors='pt'
        )
        
        output = self.model.generate(
            input_ids, 
            max_length=100, 
            num_return_sequences=1,
            temperature=0.7
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def augment_dataset(self, input_file: str, output_file: str, augmentation_factor: int = 2):
        """Augment dataset by generating alternative questions."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                original_dataset = json.load(f)
            
            augmented_dataset = []
            
            for entry in original_dataset:
                # Keep original entry
                augmented_dataset.append(entry)
                
                # Generate alternative variations
                for _ in range(augmentation_factor):
                    new_entry = entry.copy()
                    
                    # Slight variations in question and answer
                    try:
                        alt_question = self.generate_alternative_question(entry['question'])
                        new_entry['question'] = alt_question
                    except Exception as e:
                        self.logger.warning(f"Question generation failed: {e}")
                    
                    augmented_dataset.append(new_entry)
            
            # Shuffle to mix original and augmented entries
            random.shuffle(augmented_dataset)
            
            # Save augmented dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(augmented_dataset, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Augmented dataset saved to {output_file}")
            return len(augmented_dataset)
        
        except Exception as e:
            self.logger.error(f"Dataset augmentation failed: {e}")
            raise

def main():
    enhancer = QADatasetEnhancer()
    enhancer.augment_dataset(
        input_file=r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\qa_pairs.json', 
        output_file=r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\augmented_qa_pairs.json',
        augmentation_factor=2  # Generate 2 alternative entries per original
    )

if __name__ == "__main__":
    main()