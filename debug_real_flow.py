#!/usr/bin/env python3
"""Debug script to test the real flow with OpenAI API."""

import asyncio
import os
import json
import re
from openai import OpenAI
from app.rag_providers import RAGProviderRegistry
from app.rag_providers.docs_provider import docs_provider  # This triggers registration
from app.config import DOCS_BASE_URLS
from app.path_registry import path_registry

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Base template from chat.py
BASE_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology and Aptos Move code. Your task is to provide accurate, technical explanations based on the following documentation context:

Context:
{context}

When answering developer questions:

1. ACCURACY FIRST: 
   - Only answer based on information provided in the context
   - Clearly identify when you're uncertain or when information is missing
   - Never hallucinate features, functions, or capabilities not explicitly mentioned in the documentation

2. QUESTION UNDERSTANDING:
   - Begin by identifying the core intent behind the developer's question
   - Recognize when questions have multiple parts or implied sub-questions

3. CITING SOURCES:
   - Always identify the specific document source for your information
   - When a section in the context includes a URL (format: "Section: Title - URL"), you MUST use that EXACT URL for citations - DO NOT modify or construct your own URLs
   - If no URL is provided in the context for a section, do not create a URL
   - Format citations like: [Document Title](full_url_to_document)
   - NEVER construct URLs manually - only use URLs that are explicitly provided in the context

4. TECHNICAL PRECISION:
   - Use exact technical terminology from the Aptos documentation
   - Maintain precise technical meanings - don't simplify at the expense of accuracy

5. CODE EXAMPLES:
   - Provide complete, working code examples when relevant
   - Always specify the language with code blocks: ```move, ```typescript, ```python, etc.

6. FORMATTING:
   - Use proper markdown formatting for readability
   - Structure responses with clear headings and subheadings

7. LINKS AND REFERENCES:
   - When a section in the context includes a URL (format: "Section: Title - URL"), you MUST use that EXACT URL for citations
   - DO NOT modify, construct, or create your own URLs - only use URLs that are explicitly provided in the context sections
   - If no URL is provided in a context section, do not create or guess a URL

Remember: You are supporting Aptos developers who need accurate technical information. Prioritize precision and correctness over simplification.
"""

async def debug_real_flow():
    """Debug the real flow with actual RAG retrieval and OpenAI API."""
    
    print("=== Debugging Real Flow with OpenAI API ===")
    
    # Test question
    question = "How can I query fungible assets and coin balance?"
    print(f"Question: {question}")
    
    # Initialize RAG provider
    print("\n1. Initializing RAG provider...")
    try:
        rag_provider = RAGProviderRegistry.get_provider("docs")
        await rag_provider.switch_provider("developer-docs")
        print("✅ RAG provider initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG provider: {e}")
        return
    
    # Get relevant context
    print("\n2. Retrieving relevant context...")
    try:
        context_chunks = await rag_provider.get_relevant_context(
            question,
            k=5,
            include_series=True,
            provider_type="developer-docs",
            use_multi_step=False
        )
        print(f"✅ Retrieved {len(context_chunks)} context chunks")
    except Exception as e:
        print(f"❌ Failed to retrieve context: {e}")
        return
    
    # Initialize path registry
    print("\n3. Initializing path registry...")
    await path_registry.initialize_from_config('data/generated/developer-docs/url_mappings.yaml')
    print(f"✅ Path registry initialized with {len(path_registry.get_all_urls())} URLs")
    
    # Format context using the same logic as chat.py
    print("\n4. Formatting context...")
    base_url = DOCS_BASE_URLS['developer-docs']
    formatted_context = ""
    
    print(f"Base URL: {base_url}")
    
    for i, chunk in enumerate(context_chunks):
        print(f"\n--- Processing Chunk {i+1} ---")
        source_path = chunk.get('source', '')
        section_title = chunk.get('title', '')  # Use title instead of section
        content = chunk.get('content', '')
        
        print(f"Source: '{source_path}'")
        print(f"Title: '{section_title}'")
        print(f"Content preview: {content[:100]}...")
        
        if source_path:
            # Apply the same logic as in chat.py
            malformed_pattern = "data/developer-docs/apps/nextra/pages/en/"
            
            if malformed_pattern in source_path:
                print("  -> Detected malformed path, correcting...")
                # Extract the correct relative path from the last /en/ occurrence
                en_index = source_path.rfind("/en/")
                if en_index != -1:
                    corrected_path = "en/" + source_path[en_index + 4:]
                    print(f"  -> Corrected path: '{corrected_path}'")
                    
                    # Try to get URL with corrected path
                    chunk_url = path_registry.get_url(corrected_path)
                    print(f"  -> Registry lookup: '{chunk_url}'")
                    
                    if chunk_url:
                        # Add the base URL to create full URL
                        full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                        
                        # Extract section title from content if title field is empty
                        if not section_title:
                            # Try to extract from content pattern like "Context: ... > Section Title"
                            content = chunk.get('content', '')
                            if content.startswith('Context:'):
                                # Extract the last part after the last '>'
                                lines = content.split('\n')
                                if lines:
                                    first_line = lines[0]
                                    if '>' in first_line:
                                        section_title = first_line.split('>')[-1].strip()
                                    else:
                                        # If no '>', take everything after "Context: "
                                        section_title = first_line.replace('Context:', '').strip()
                                print(f"  -> Extracted section title from content: '{section_title}'")
                        
                        # Add anchor based on section title
                        if section_title:
                            # Convert section title to proper URL anchor format
                            anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
                            # Remove any remaining special characters and multiple dashes
                            anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
                            anchor = re.sub(r'-+', '-', anchor).strip('-')
                            if anchor:
                                full_url += f"#{anchor}"
                                print(f"  -> Added anchor: '{anchor}'")
                        
                        print(f"  -> Final URL: '{full_url}'")
                        formatted_context += f"\n\nSection: {section_title} - {full_url}\n"
                    else:
                        print("  -> No URL found in registry")
                        formatted_context += f"\n\nSection: {section_title}\n"
                else:
                    print("  -> Could not find /en/ in path")
                    formatted_context += f"\n\nSection: {section_title}\n"
            else:
                print("  -> Using normal URL lookup")
                # Try normal URL lookup
                chunk_url = path_registry.get_url(source_path)
                print(f"  -> Registry lookup: '{chunk_url}'")
                
                if chunk_url:
                    # Add the base URL to create full URL
                    full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                    
                    # Add anchor based on section title
                    if section_title:
                        # Convert section title to proper URL anchor format
                        anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
                        # Remove any remaining special characters and multiple dashes
                        anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
                        anchor = re.sub(r'-+', '-', anchor).strip('-')
                        if anchor:
                            full_url += f"#{anchor}"
                            print(f"  -> Added anchor: '{anchor}'")
                    
                    print(f"  -> Final URL: '{full_url}'")
                    formatted_context += f"\n\nSection: {section_title} - {full_url}\n"
                else:
                    print("  -> No URL found in registry")
                    formatted_context += f"\n\nSection: {section_title}\n"
        else:
            formatted_context += f"\n\nSection: {section_title}\n"
        
        if chunk.get("summary"):
            formatted_context += f"Summary: {chunk.get('summary')}\n"
        formatted_context += f"Content: {content}\n"
    
    print(f"\n5. Final formatted context:")
    print("=" * 80)
    print(formatted_context)
    print("=" * 80)
    
    # Prepare the prompt
    print("\n6. Preparing prompt for OpenAI...")
    prompt = BASE_TEMPLATE.format(
        context=formatted_context,
        question=question,
        main_topic="fungible assets"
    )
    
    # Call OpenAI API
    print("\n7. Calling OpenAI API...")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.05
        )
        
        ai_response = response.choices[0].message.content
        print("✅ OpenAI API call successful")
        
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return
    
    print(f"\n8. AI Response:")
    print("=" * 80)
    print(ai_response)
    print("=" * 80)
    
    # Analyze the response for URLs
    print(f"\n9. URL Analysis:")
    
    # Extract URLs from the response
    url_pattern = r'https://aptos\.dev[^\s\)]*'
    found_urls = re.findall(url_pattern, ai_response)
    
    print(f"URLs found in AI response: {len(found_urls)}")
    for i, url in enumerate(found_urls, 1):
        print(f"  {i}. {url}")
    
    # Check if any URLs have incorrect anchors
    incorrect_urls = []
    for url in found_urls:
        if '#build' in url or '#buildguides' in url or '#guides' in url:
            incorrect_urls.append(url)
    
    if incorrect_urls:
        print(f"\n❌ Found {len(incorrect_urls)} URLs with incorrect anchors:")
        for url in incorrect_urls:
            print(f"  - {url}")
    else:
        print(f"\n✅ No URLs with incorrect anchors found")
    
    # Check if AI is using the URLs we provided
    context_urls = re.findall(url_pattern, formatted_context)
    print(f"\nURLs provided in context: {len(context_urls)}")
    for i, url in enumerate(context_urls, 1):
        print(f"  {i}. {url}")
    
    # Check if AI used our URLs vs created new ones
    ai_used_our_urls = all(url in formatted_context for url in found_urls)
    print(f"\nDid AI use only our provided URLs? {'✅ Yes' if ai_used_our_urls else '❌ No'}")
    
    if not ai_used_our_urls:
        print("AI created new URLs not in our context:")
        for url in found_urls:
            if url not in formatted_context:
                print(f"  - {url}")

if __name__ == "__main__":
    asyncio.run(debug_real_flow()) 