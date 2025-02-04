import json
from inference.inference import PolicyQASystem

def main():
    qa_system = PolicyQASystem(
        data_path=r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\qa_pairs.json'
    )
    
    print("✅ Policy Q&A system loaded. You can now ask questions.")
    
    while True:
        query = input("Ask a policy-related question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting...")
            break
        response = qa_system.get_answer(query)
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    main()