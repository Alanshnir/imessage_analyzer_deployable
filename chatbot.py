"""
GPT-Powered Texting Behavior Chatbot (RQ9).

This module provides a chatbot interface that answers questions about texting patterns
using ONLY aggregated statistics from RQ1-8.

PRIVACY: NO raw message texts, phone numbers, or personally identifiable information
are sent to the OpenAI API. Only aggregated statistics and patterns are included.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json

# Import OpenAI at module level so we can see real errors
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def build_behavior_summary(rq_results: Dict[str, Any]) -> str:
    """
    Build a comprehensive behavior summary from RQ1-8 results.
    
    PRIVACY: This function ONLY includes aggregated statistics.
    NO raw messages, phone numbers, emails, or PII are included.
    
    Args:
        rq_results: Dictionary containing results from all research questions
        
    Returns:
        Text summary of aggregated texting behavior patterns
    """
    summary_parts = []
    
    summary_parts.append("# User's iMessage Behavior Summary (Aggregated Statistics Only)")
    summary_parts.append("\nThis summary contains ONLY aggregated patterns and statistics.")
    summary_parts.append("NO raw messages or personally identifiable information is included.\n")
    
    # RQ1: Topics User Tends Not to Engage With
    if 'rq1_df' in rq_results and rq_results['rq1_df'] is not None:
        rq1_df = rq_results['rq1_df']
        num_rq1_topics = len(rq1_df)
        summary_parts.append(f"\n## RQ1: Topics User Tends Not to Engage With ({num_rq1_topics} Topics)")
        
        if num_rq1_topics > 0:
            summary_parts.append(f"- Total topics analyzed: {num_rq1_topics}")
            summary_parts.append("- All topics ranked by reluctance:")
            
            for _, row in rq1_df.iterrows():
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} "
                    f"(Reluctance: {row['reluctance_score']:.3f}, Frequency: {row['frequency']} messages, "
                    f"Rank Score: {row['final_rank']:.3f})"
                )
    
    # RQ2: Most Commonly Discussed Topics
    if 'rq2_topic_words' in rq_results and rq_results['rq2_topic_words'] is not None:
        rq2_df = rq_results['rq2_topic_words']
        num_rq2_topics = len(rq2_df)
        summary_parts.append(f"\n## RQ2: Most Commonly Discussed Topics ({num_rq2_topics} Topics)")
        
        if num_rq2_topics > 0:
            # Sort by message count
            rq2_sorted = rq2_df.sort_values('message_count', ascending=False)
            summary_parts.append(f"- Total topics: {num_rq2_topics}")
            summary_parts.append("- All topics ranked by frequency:")
            
            for _, row in rq2_sorted.iterrows():
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} "
                    f"({row['message_count']} messages)"
                )
    
    # RQ2 Per-Contact Topic Mix
    if 'rq2_contact_topic' in rq_results and rq_results['rq2_contact_topic'] is not None:
        contact_topic_df = rq_results['rq2_contact_topic']
        if len(contact_topic_df) > 0:
            summary_parts.append("\n- Per-contact topic preferences (anonymized):")
            for idx, row in contact_topic_df.iterrows():
                # Get top 3 topics for this contact
                topic_cols = [col for col in row.index if str(col).startswith('topic_')]
                if topic_cols:
                    top_topics = sorted([(col, row[col]) for col in topic_cols], 
                                       key=lambda x: x[1], reverse=True)[:3]
                    contact_label = f"Contact_{chr(65 + idx % 26)}"
                    topic_desc = ', '.join([f"T{col.split('_')[1]}: {val:.1%}" 
                                           for col, val in top_topics])
                    summary_parts.append(f"  * {contact_label}: {topic_desc}")
    
    # RQ3: Group vs One-to-One Responsiveness
    if 'rq3_stats' in rq_results and rq_results['rq3_stats'] is not None:
        rq3_stats = rq_results['rq3_stats']
        summary_parts.append("\n## RQ3: Group vs One-to-One Responsiveness")
        
        if 'by_category' in rq3_stats:
            by_cat = rq3_stats['by_category']
            summary_parts.append("- Reply rates by conversation type:")
            for _, row in by_cat.iterrows():
                summary_parts.append(
                    f"  * {row['category']}: {row['reply_rate']:.1%} reply rate, "
                    f"median response: {row['median_response_time']:.1f} min, "
                    f"mean response: {row['mean_response_time']:.1f} min, "
                    f"{row['count']} messages"
                )
        
        if 'contact_comparison' in rq3_stats:
            contact_comp = rq3_stats['contact_comparison'].copy()
            contact_comp['reply_diff'] = contact_comp['group_reply_rate'] - contact_comp['one_on_one_reply_rate']
            contact_comp['total_msgs'] = contact_comp['one_on_one_count'] + contact_comp['group_count']
            
            # Sort by reply difference
            contact_comp_sorted = contact_comp.sort_values('reply_diff', ascending=False)
            
            summary_parts.append(f"- Contact-level reply behavior (total: {len(contact_comp_sorted)} contacts):")
            summary_parts.append("  Contacts anonymized as Contact_A, Contact_B, etc.")
            
            # Show all contacts with their reply differences
            for idx, row in contact_comp_sorted.iterrows():
                summary_parts.append(
                    f"  * Contact_{chr(65 + idx % 26)}: "
                    f"1-on-1 reply rate: {row['one_on_one_reply_rate']:.1%}, "
                    f"Group reply rate: {row['group_reply_rate']:.1%}, "
                    f"Difference: {row['reply_diff']:+.1%}, "
                    f"Total msgs: {int(row['total_msgs'])}"
                )
    
    # RQ4: Conversation Starter Topics
    if 'rq4_starters' in rq_results and rq_results['rq4_starters'] is not None:
        rq4_df = rq_results['rq4_starters']
        num_rq4_topics = len(rq4_df)
        summary_parts.append(f"\n## RQ4: Conversation Starter Topics (Fine-Grained, {num_rq4_topics} Topics)")
        
        if num_rq4_topics > 0:
            summary_parts.append(f"- Total starter topics analyzed: {num_rq4_topics}")
            summary_parts.append("- All conversation starter topics ranked:")
            
            for _, row in rq4_df.iterrows():
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} "
                    f"(Starter Score: {row['starter_score']:.3f}, Reply Rate: {row['reply_rate']:.1%}, "
                    f"Avg Response: {row['avg_response_time']:.1f} min, Starter Prob: {row['starter_probability']:.1%})"
                )
    
    # RQ5: Conversation Ender Topics
    if 'rq5_enders' in rq_results and rq_results['rq5_enders'] is not None:
        rq5_df = rq_results['rq5_enders']
        num_rq5_topics = len(rq5_df)
        summary_parts.append(f"\n## RQ5: Conversation Ender Topics (Fine-Grained, {num_rq5_topics} Topics)")
        
        if num_rq5_topics > 0:
            summary_parts.append(f"- Total ender topics analyzed: {num_rq5_topics}")
            summary_parts.append("- All conversation ender topics ranked:")
            
            for _, row in rq5_df.iterrows():
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} "
                    f"(Ender Score: {row['ender_score']:.3f}, No-Reply Rate: {row['no_reply_rate']:.1%}, "
                    f"Avg Response: {row['avg_response_time']:.1f} min, Ender Prob: {row['ender_probability']:.1%})"
                )
    
    # RQ6: Topics by Closeness
    if 'rq6_closeness' in rq_results and rq_results['rq6_closeness'] is not None:
        rq6_df = rq_results['rq6_closeness']
        num_rq6_topics = len(rq6_df)
        summary_parts.append(f"\n## RQ6: Topics by Closeness ({num_rq6_topics} Topics)")
        summary_parts.append("Comparing topic prevalence between close contacts vs acquaintances")
        
        if num_rq6_topics > 0:
            summary_parts.append(f"- Total topics analyzed: {num_rq6_topics}")
            summary_parts.append("- All topics with closeness data:")
            
            for _, row in rq6_df.iterrows():
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} "
                    f"(Category: {row['category']}, Odds Ratio: {row['odds_ratio']:.2f}x, "
                    f"Close Freq: {row['close_frequency']:.1%}, Acquaint Freq: {row['acquaintance_frequency']:.1%})"
                )
    
    # RQ7: Topics by Time of Day
    if 'rq7_time' in rq_results and rq_results['rq7_time'] is not None:
        rq7_df = rq_results['rq7_time']
        num_rq7_topics = len(rq7_df)
        summary_parts.append(f"\n## RQ7: Topics by Time of Day ({num_rq7_topics} Topics)")
        summary_parts.append("Time periods: Morning (6AM-12PM), Afternoon (12PM-6PM), Evening (6PM-10PM), Night (10PM-6AM)")
        
        if num_rq7_topics > 0:
            summary_parts.append(f"- Total topics analyzed: {num_rq7_topics}")
            summary_parts.append("- All topics with time-of-day frequencies:")
            
            time_periods = ['morning', 'afternoon', 'evening', 'night']
            for _, row in rq7_df.iterrows():
                time_data = ', '.join([f"{period.capitalize()}: {row[period]:.1%}" 
                                      for period in time_periods if period in row])
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} ({time_data})"
                )
    
    # RQ8: Fine-Grained Avoided Topics
    if 'rq8_df' in rq_results and rq_results['rq8_df'] is not None:
        rq8_df = rq_results['rq8_df']
        num_rq8_topics = len(rq8_df)
        summary_parts.append(f"\n## RQ8: Fine-Grained Topics User Tends Not to Engage With ({num_rq8_topics} Topics, TF-IDF + K-Means)")
        
        if num_rq8_topics > 0:
            summary_parts.append(f"- Total fine-grained topics: {num_rq8_topics}")
            summary_parts.append("- All fine-grained avoided topics ranked:")
            
            for _, row in rq8_df.iterrows():
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {row['top_words']} "
                    f"(Reluctance: {row['reluctance_score']:.3f}, Frequency: {row['frequency']} messages, "
                    f"Rank Score: {row['final_rank']:.3f})"
                )
    
    # Add overall statistics
    if 'stats' in rq_results:
        stats = rq_results['stats']
        summary_parts.append("\n## Overall Statistics")
        summary_parts.append(f"- Total messages analyzed: {stats.get('total_messages', 'N/A')}")
        summary_parts.append(f"- Messages received: {stats.get('messages_received', 'N/A')}")
        summary_parts.append(f"- Messages sent: {stats.get('messages_sent', 'N/A')}")
        summary_parts.append(f"- Overall reply rate: {stats.get('reply_rate', 0):.1%}")
        summary_parts.append(f"- Median response time: {stats.get('median_response_time', 'N/A'):.1f} minutes")
    
    summary_parts.append("\n---")
    summary_parts.append("\nREMINDER: Base your answers ONLY on these aggregated statistics.")
    summary_parts.append("You do NOT have access to raw messages, contact names, or any PII.")
    
    return '\n'.join(summary_parts)


def run_rq9_chatbot(
    user_question: str,
    behavior_summary: str,
    api_key: str,
    conversation_history: list = None
) -> tuple[str, list]:
    """
    Run the GPT-powered chatbot to answer questions about texting behavior.
    
    PRIVACY: This function only sends aggregated statistics to the OpenAI API.
    NO raw messages, phone numbers, or personally identifiable information is sent.
    
    Args:
        user_question: The user's question
        behavior_summary: Aggregated behavior summary from build_behavior_summary()
        api_key: OpenAI API key
        conversation_history: Previous conversation messages (optional)
        
    Returns:
        Tuple of (assistant response, updated conversation history)
    """
    if not OPENAI_AVAILABLE or OpenAI is None:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Build conversation messages
    messages = []
    
    # System prompt with behavior summary
    system_prompt = f"""You are a helpful assistant that interprets a user's iMessage texting behavior.

You have access ONLY to aggregated statistics and patterns from their texting analysis.
You do NOT have access to raw messages, contact names, phone numbers, or any personally identifiable information.

Base ALL your answers strictly on the aggregated patterns provided below.

If the user asks about specific messages or contacts, explain that you only have access to aggregated patterns, not individual messages.

AGGREGATED BEHAVIOR PATTERNS:
{behavior_summary}

When answering:
- Be insightful and analytical
- Use the specific metrics and patterns provided
- Avoid making up information not in the summary
- If a pattern isn't in the data, say so
- Be friendly and conversational"""

    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user question
    messages.append({"role": "user", "content": user_question})
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",  # Using gpt-4o (latest, more cost-effective than gpt-4-turbo)
        messages=messages,
        temperature=0.7,
        max_tokens=2000  # Increased for more detailed responses with full data
    )
    
    # Extract assistant response
    assistant_message = response.choices[0].message.content
    
    # Update conversation history
    updated_history = conversation_history.copy() if conversation_history else []
    updated_history.append({"role": "user", "content": user_question})
    updated_history.append({"role": "assistant", "content": assistant_message})
    
    # Keep only last 10 messages to avoid context overflow
    if len(updated_history) > 20:  # 10 exchanges = 20 messages
        updated_history = updated_history[-20:]
    
    return assistant_message, updated_history


def get_example_questions() -> list:
    """Get example questions users can ask the chatbot."""
    return [
        "What topics do I reply to the fastest?",
        "Why do I avoid certain topics?",
        "How do I text differently with close friends vs acquaintances?",
        "Do I end conversations in specific ways?",
        "Which topics do I usually start conversations with?",
        "What time of day am I most responsive?",
        "Are there topics I only discuss in groups?",
        "What patterns show I'm reluctant to engage?",
        "Compare my behavior in group chats vs one-on-one",
        "What topics get no response from me?"
    ]

