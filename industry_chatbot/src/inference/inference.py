import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class PolicyQASystem:
    def __init__(self, data_path, model_name="EleutherAI/gpt-neo-125M"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        self.retrieval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.gen_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.questions = [item["question"] for item in self.data]
        self.answers = {q: item["answer"] for q, item in zip(self.questions, self.data)}
        self._build_index()
        
        # Context-specific templates
        self.templates = {
            'customer_satisfaction': [
                "Our customer satisfaction policy focuses on delivering services {} while ensuring {}",
                "We are committed to achieving complete customer satisfaction {} through {}",
                "To ensure customer satisfaction, we deliver our services {} and leverage {}"
            ],
            'default': [
                "Our policy establishes that {}",
                "According to our policy, {}",
                "The policy specifies that {}"
            ]
        }
        
    def _build_index(self):
        self.question_embeddings = self.retrieval_model.encode(
            self.questions, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        self.index = faiss.IndexFlatL2(self.question_embeddings.shape[1])
        self.index.add(self.question_embeddings)
    
    def _format_policy_answer(self, answer, question):
        """Format the answer based on question context and policy type"""
        try:
            # Clean the answer
            answer = answer.strip()
            if answer.endswith('.'):
                answer = answer[:-1]
                
            # Identify policy type from question
            question_lower = question.lower()
            
            if 'customer satisfaction' in question_lower:
                # Parse cost and technology components
                if 'and' in answer:
                    cost_part, tech_part = answer.lower().split('and')
                    # Format specifically for customer satisfaction policy
                    templates = self.templates['customer_satisfaction']
                    template = np.random.choice(templates)
                    return template.format(cost_part.strip(), tech_part.strip())
            
            # Default formatting for other policy types
            templates = self.templates['default']
            template = np.random.choice(templates)
            return template.format(answer)
            
        except Exception as e:
            print(f"Formatting failed: {e}")
            return answer

    def get_answer(self, query):
        # Get closest matching answer
        query_embedding = self.retrieval_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k=1)
        
        if indices[0][0] >= len(self.questions):
            return "I don't have information on that topic."
            
        matched_question = self.questions[indices[0][0]]
        original_answer = self.answers[matched_question]
        
        # Format the answer based on context
        try:
            formatted_answer = self._format_policy_answer(original_answer, query)
            return formatted_answer
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return original_answer

if __name__ == "__main__":
    qa_system = PolicyQASystem(
        data_path=r"C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\qa_pairs.json"
    )
    query = "Explain network documentation policy according to the IT policy"
    response = qa_system.get_answer(query)
    print(f"Query: {query}\nEnhanced Answer: {response}")