import re

def clean_heading(heading):
    """
    Clean the heading text by removing unwanted punctuation and applying replacements.
    """
    heading = heading.strip()
    heading = re.sub(r'[\:\-]+$', '', heading)  # Remove trailing colons, dashes
    heading = heading.replace("Policy Ref. No.", "Policy Reference")  # Example standardization
    return heading

def clean_content(content):
    """
    Clean the content text by removing irrelevant details and normalizing spaces.
    """
    content = re.sub(r'Sheet No\.:.*\n?', '', content)  # Remove sheet details
    content = re.sub(r'Page \d+', '', content)  # Remove page numbers
    content = re.sub(r'\s+', ' ', content)  # Normalize spaces
    return content.strip()

def generate_questions(heading):
    """
    Generate different question variations based on the heading.
    """
    heading_norm = heading.lower()
    return [
        f"What is the {heading_norm} policy?",
        f"Explain {heading_norm} according to the IT policy",
        f"What are the rules for {heading_norm}?"
    ]
