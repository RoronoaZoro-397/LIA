"""
RAG Document Assistant Bot for RingCentral

This bot integrates the Streamlit RAG functionality into RingCentral
to allow users to query their Google Drive documents through chat.
"""
__name__ = 'localConfig'
__package__ = 'ringcentral_bot_framework'

import copy
import requests
import os
import json
import hashlib
import glob
import pickle
import time
from datetime import datetime
import tempfile
from typing import Dict, List, Optional, Any

# Import the necessary functions from your Streamlit app
from streamlit_app_drive import (
    create_opensearch_client,
    get_document_count,
    load_enhanced_file_metadata,
    classify_query,
    handle_metadata_query,
    UnifiedRetriever,
    initialize_llamaindex_llm,
    initialize_llamaindex_embeddings,
    query_unified_system,
    initialize_unified_retriever_from_existing,
    get_google_oauth_url,
    authenticate_with_google,
    process_all_user_documents_unified
)

# Import spaCy metadata handler
from metadata_handler import get_spacy_metadata_handler

# Import image processor
from image_processor import init_gemini, process_pdf_images

# Import status updater
from status_updater import process_query_with_simple_status, process_documents_with_simple_status

# Import Google OAuth and Drive functionality
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
OPENSEARCH_URL = os.getenv('OPENSEARCH_URL', 'http://localhost:9200')
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME', '')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'document_embeddings')
SHEETS_INDEX = 'sheets_embeddings'

# Google OAuth configuration
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

class RAGBotState:
    """Manages bot state and user sessions"""
    
    def __init__(self):
        self.user_sessions = {}  # Store user-specific data
        self.unified_retriever = None
        self.llm = None
        self.initialized = False
        self.oauth_flows = {}  # Store OAuth flows per user
    
    def get_user_session(self, user_id: str) -> Dict:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            # Try to load saved auth state
            saved_auth = load_auth_state(user_id)
            
            self.user_sessions[user_id] = {
                'authenticated': saved_auth.get('authenticated', False),
                'credentials': None,  # Don't save credentials for security
                'user_info': saved_auth.get('user_info'),
                'chat_history': [],
                'processed_documents': saved_auth.get('processed_documents', False),
                'user_email': saved_auth.get('user_email'),
                'auth_step': saved_auth.get('auth_step', 'not_started'),
                'oauth_flow': None
            }
            
            # If user was previously authenticated and documents were processed, 
            # check if documents still exist
            if (saved_auth.get('authenticated', False) and 
                saved_auth.get('processed_documents', False)):
                try:
                    pdf_count = get_document_count(OPENSEARCH_INDEX)
                    sheets_count = get_document_count(SHEETS_INDEX)
                    if pdf_count > 0 or sheets_count > 0:
                        print(f"✅ Auto-restored auth for user {user_id} - documents available")
                    else:
                        # Documents were cleared, reset auth state
                        self.user_sessions[user_id]['authenticated'] = False
                        self.user_sessions[user_id]['processed_documents'] = False
                        self.user_sessions[user_id]['auth_step'] = 'not_started'
                        print(f"⚠️ Auth restored but no documents found for user {user_id}")
                except Exception as e:
                    print(f"⚠️ Error checking documents for user {user_id}: {e}")
                    # Reset auth state if we can't verify documents
                    self.user_sessions[user_id]['authenticated'] = False
                    self.user_sessions[user_id]['processed_documents'] = False
                    self.user_sessions[user_id]['auth_step'] = 'not_started'
        
        return self.user_sessions[user_id]
    
    def initialize_components(self):
        """Initialize components using the same logic as Streamlit app"""
        if not self.initialized:
            try:
                # Use the exact same initialization as Streamlit app
                if initialize_unified_retriever_from_existing():
                    self.unified_retriever = initialize_unified_retriever_from_existing()
                    self.llm = initialize_llamaindex_llm()
                    self.initialized = True
                    print("✅ RAG components initialized successfully using Streamlit backend")
                else:
                    print("❌ Failed to initialize RAG components")
                    
            except Exception as e:
                print(f"❌ Error initializing components: {str(e)}")

# Global bot state
rag_bot_state = RAGBotState()

def save_auth_state(user_id: str, user_session: Dict):
    """Save authentication state to disk"""
    try:
        auth_dir = "./auth_states"
        os.makedirs(auth_dir, exist_ok=True)
        
        # Save only essential auth data (not sensitive credentials)
        auth_data = {
            'user_id': user_id,
            'authenticated': user_session.get('authenticated', False),
            'user_email': user_session.get('user_email'),
            'user_info': user_session.get('user_info'),
            'processed_documents': user_session.get('processed_documents', False),
            'auth_step': user_session.get('auth_step', 'not_started'),
            'saved_at': datetime.now().isoformat()
        }
        
        auth_file = os.path.join(auth_dir, f"auth_{user_id}.pkl")
        with open(auth_file, 'wb') as f:
            pickle.dump(auth_data, f)
        
        print(f"✅ Saved auth state for user {user_id}")
    except Exception as e:
        print(f"❌ Error saving auth state: {e}")

def load_auth_state(user_id: str) -> Dict:
    """Load authentication state from disk"""
    try:
        auth_dir = "./auth_states"
        auth_file = os.path.join(auth_dir, f"auth_{user_id}.pkl")
        
        if os.path.exists(auth_file):
            with open(auth_file, 'rb') as f:
                auth_data = pickle.load(f)
            
            print(f"✅ Loaded auth state for user {user_id}")
            return auth_data
        else:
            return {}
    except Exception as e:
        print(f"❌ Error loading auth state: {e}")
        return {}

def clear_auth_state(user_id: str):
    """Clear authentication state from disk"""
    try:
        auth_dir = "./auth_states"
        auth_file = os.path.join(auth_dir, f"auth_{user_id}.pkl")
        
        if os.path.exists(auth_file):
            os.remove(auth_file)
            print(f"✅ Cleared auth state for user {user_id}")
    except Exception as e:
        print(f"❌ Error clearing auth state: {e}")

def botJoinPrivateChatAction(bot, groupId, user, dbAction):
    """
    This is invoked when the bot is added to a private group.
    """
    bot.sendMessage(
        groupId,
        {
            'text': 
            f'''
            **Hello! I am LIA. **
            
            I can help you query your Google Drive documents, PDFs, and spreadsheets!
            
            **Setup Required:**
            1. **Authenticate with Google** - Type "!auth" to start (only needed once)
            2. **Process Documents** - Type "!process" after authentication (only needed once)
            3. **Start Querying** - Ask questions about your documents
            
            **Note:** If you were previously authenticated and documents are processed, you can start chatting directly!
            
            **Other Commands:**
            - `!status` - Check current status
            - `!reset` - Clear all data and start fresh
            - `!clear` - Clear chat history
            
            **How to use me:**
            - Type "!auth" to authenticate with Google Drive
            - Type "!process" to scan and process your documents
            - Ask questions: "What does the Q4 report say about revenue?"
            - Find content: "Find files containing budget information"
            
            **Examples:**
            - "What are the key points in the marketing strategy document?"
            - "Find files containing quarterly results"
            - "Show me the most recent PDF files"
            - "What does the budget spreadsheet say about expenses?"
            
            Reply ![:Person]({bot.id}) if you need help or want to see these instructions again.
            '''
        }
    )

def botGotPostAddAction(
    bot,
    groupId,
    creatorId,
    user,
    text,
    dbAction,
    handledByExtension,
    event
):
    """
    This is invoked when the user sends a message to the bot.
    """
    if handledByExtension:
        return

    # Get user session
    user_session = rag_bot_state.get_user_session(creatorId)
    
    # Check if user is asking for help
    if f'![:Person]({bot.id})' in text:
        print(f"🔍 User asked for help: {text}")
        bot.sendMessage(
            groupId,
            {
                'text': 
                f'''
                **Hello I am LIA your Document Assistant Bot **
                
                **Setup Commands:**
                - `!auth` - Authenticate with Google Drive
                - `!process` - Process your documents (after auth)
                - `!status` - Check current status
                - `!reset` - Clear all data and start fresh
                - `!clear` - Clear chat history
                
                **What I can do:**
                - Answer questions about your documents and spreadsheets
                - Find specific content across your files
                - Provide file metadata and statistics
                - Search for topics and keywords
                
                **Example queries:**
                - "What does the Q4 report say about revenue growth?"
                - "Find files containing budget information"
                - "Show me the largest PDF files from last month"
                - "What documents discuss AI implementation?"
                - "Summarize the key points from the marketing strategy"
                
                **Just ask me anything about your documents!**
                '''
            }
        )
        return
    
    # Handle authentication command
    if text.lower() == "!auth":
        handle_auth_command(bot, groupId, creatorId, user_session)
        return
    
    # Handle processing command
    if text.lower() == "!process":
        handle_process_command(bot, groupId, creatorId, user_session)
        return
    
    # Handle status command
    if text.lower() == "!status":
        handle_status_command(bot, groupId, creatorId, user_session)
        return
    
    # Handle reset command
    if text.lower() == "!reset":
        handle_reset_command(bot, groupId, creatorId, user_session)
        return
    
    # Handle clear chat command
    if text.lower() == "!clear":
        handle_clear_chat_command(bot, groupId, creatorId, user_session)
        return
    
    # Handle OAuth callback
    if user_session['auth_step'] == 'waiting_auth' and len(text) > 10 and 'http' not in text:
        handle_oauth_callback(bot, groupId, creatorId, user_session, text)
        return
    
    # Check if user is authenticated and documents are processed
    if not user_session['authenticated']:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), please authenticate first by typing "!auth"'
            }
        )
        return
    
    if not user_session['processed_documents']:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), please process your documents first by typing "!process"'
            }
        )
        return
    
    # Initialize components if not done yet
    if not rag_bot_state.initialized:
        rag_bot_state.initialize_components()
    
    # Check if we have the necessary components
    if not rag_bot_state.unified_retriever or not rag_bot_state.llm:
        # Try to initialize components
        if not rag_bot_state.initialized:
            rag_bot_state.initialize_components()
        
        # Check again after initialization attempt
        if not rag_bot_state.unified_retriever or not rag_bot_state.llm:
            bot.sendMessage(
                groupId,
                {
                    'text': f'![:Person]({creatorId}), I\'m sorry, but the document processing system is not ready yet. Please ensure that documents have been processed and the system is properly configured.'
                }
            )
            return
    
    # Process the user's query with dynamic status updates
    try:
        # Add user message to chat history
        user_session['chat_history'].append({
            "role": "user",
            "content": text
        })
        
        # Limit chat history to last 10 messages to avoid token limits
        if len(user_session['chat_history']) > 10:
            user_session['chat_history'] = user_session['chat_history'][-10:]
        
        # Get user email for metadata queries
        user_email = user_session.get('user_email', 'unknown')
        
        # Process query with status updates
        response, document_links = process_query_with_simple_status(
            bot, groupId, creatorId, text, user_session, rag_bot_state
        )
        
    except Exception as e:
        error_message = f"❌ Error processing your query: {str(e)}"
        print(f"Error in RAG bot: {str(e)}")
        
        # Send error message
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), {error_message}'
            }
        )

def handle_auth_command(bot, groupId, creatorId, user_session):
    """Handle Google OAuth authentication"""
    try:
        # Generate OAuth URL using your Streamlit function
        auth_url = get_google_oauth_url()
        
        if auth_url:
            user_session['auth_step'] = 'waiting_auth'
            bot.sendMessage(
                groupId,
                {
                    'text': f'![:Person]({creatorId}), please authenticate with Google Drive:\n\n🔗 [Click here to authenticate]({auth_url})\n\nAfter authentication, you\'ll see an authorization code. Copy that code and paste it here.'
                }
            )
        else:
            bot.sendMessage(
                groupId,
                {
                    'text': f'![:Person]({creatorId}), failed to generate authentication URL. Please check your Google OAuth configuration.'
                }
            )
    except Exception as e:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), error starting authentication: {str(e)}'
            }
        )

def handle_oauth_callback(bot, groupId, creatorId, user_session, authorization_code):
    """Handle OAuth callback"""
    try:
        # Use your Streamlit authentication function
        credentials, user_info = authenticate_with_google(authorization_code)
        
        if credentials and user_info:
            user_session['credentials'] = credentials
            user_session['user_info'] = user_info
            user_session['authenticated'] = True
            user_session['user_email'] = user_info.get('email', 'unknown')
            user_session['auth_step'] = 'authenticated'
            
            # Save authentication state
            save_auth_state(creatorId, user_session)
            
            bot.sendMessage(
                groupId,
                {
                    'text': f'![:Person]({creatorId}), ✅ Successfully authenticated with Google!\n\nEmail: {user_info.get("email", "Unknown")}\nName: {user_info.get("name", "Unknown")}\n\nNow type "!process" to scan and process your documents.'
                }
            )
        else:
            user_session['auth_step'] = 'not_started'
            bot.sendMessage(
                groupId,
                {
                    'text': f'![:Person]({creatorId}), ❌ Authentication failed. Please try "!auth" again.'
                }
            )
    except Exception as e:
        user_session['auth_step'] = 'not_started'
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), ❌ Error during authentication: {str(e)}'
            }
        )

def handle_reset_command(bot, groupId, creatorId, user_session):
    """Handle database reset command"""
    try:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), 🧹 Starting database reset...\n\nThis will clear all processed documents and metadata. Please wait...'
            }
        )
        
        # 1. Remove user metadata files
        metadata_files = glob.glob("./vectorstores/user_*_metadata.json")
        removed_count = 0
        for file in metadata_files:
            try:
                os.remove(file)
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {file}: {e}")
        
        # 1.5. Clear all auth states
        auth_files = glob.glob("./auth_states/auth_*.pkl")
        auth_cleared = 0
        for file in auth_files:
            try:
                os.remove(file)
                auth_cleared += 1
            except Exception as e:
                print(f"❌ Error removing {file}: {e}")
        
        # 2. Clear OpenSearch indices
        try:
            # Get list of indices
            response = requests.get("http://localhost:9200/_cat/indices?format=json", timeout=5)
            if response.status_code == 200:
                indices = response.json()
                deleted_indices = []
                for index in indices:
                    index_name = index['index']
                    if 'multimodal_documents' in index_name or 'document_embeddings' in index_name or 'sheets_embeddings' in index_name or 'file_metadata' in index_name:
                        # Delete the index
                        delete_response = requests.delete(f"http://localhost:9200/{index_name}", timeout=5)
                        if delete_response.status_code in [200, 404]:
                            deleted_indices.append(index_name)
                        else:
                            print(f"❌ Failed to delete index {index_name}: {delete_response.status_code}")
                
                indices_deleted = len(deleted_indices)
            else:
                indices_deleted = 0
        except Exception as e:
            print(f"ℹ️  OpenSearch not running or not accessible: {e}")
            indices_deleted = 0
        
        # 3. Clear any temporary files
        temp_patterns = [
            "./*.tmp",
            "./*.cache",
            "./temp_*",
            "./tmp_*"
        ]
        
        temp_files_removed = 0
        for pattern in temp_patterns:
            temp_files = glob.glob(pattern)
            for file in temp_files:
                try:
                    os.remove(file)
                    temp_files_removed += 1
                except Exception as e:
                    pass
        
        # 4. Clear Python cache
        cache_dirs = glob.glob("./**/__pycache__", recursive=True)
        cache_cleared = 0
        for cache_dir in cache_dirs:
            try:
                import shutil
                shutil.rmtree(cache_dir)
                cache_cleared += 1
            except Exception as e:
                pass
        
        # 5. Reset user session
        user_session['authenticated'] = False
        user_session['credentials'] = None
        user_session['user_info'] = None
        user_session['chat_history'] = []
        user_session['processed_documents'] = False
        user_session['user_email'] = None
        user_session['auth_step'] = 'not_started'
        user_session['oauth_flow'] = None
        
        # Clear saved auth state
        clear_auth_state(creatorId)
        
        # 6. Reset global bot state
        rag_bot_state.unified_retriever = None
        rag_bot_state.llm = None
        rag_bot_state.initialized = False
        
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), ✅ Database reset complete!\n\n📊 Reset Summary:\n• Removed {removed_count} metadata files\n• Deleted {indices_deleted} OpenSearch indices\n• Cleared {temp_files_removed} temporary files\n• Cleared {cache_cleared} cache directories\n• Cleared {auth_cleared} auth states\n\n🔄 All processed documents and metadata have been cleared.\n\nTo start fresh:\n1. Type "!auth" to authenticate with Google Drive\n2. Type "!process" to process your documents again'
            }
        )
        
    except Exception as e:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), ❌ Error during reset: {str(e)}'
            }
        )

def handle_clear_chat_command(bot, groupId, creatorId, user_session):
    """Handle clear chat command"""
    try:
        # Clear the chat history for this user
        user_session['chat_history'] = []
        
        # Try to clear RingCentral chat messages (this is limited by API permissions)
        try:
            # Note: RingCentral API doesn't provide a direct way to delete messages
            # This would require admin permissions and specific API endpoints
            # For now, we'll just clear the bot's internal chat history
            pass
        except Exception as e:
            print(f"Note: Cannot clear RingCentral chat messages: {e}")
        
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), 🗑️ Chat history cleared!\n\nYour conversation history has been reset. You can continue asking questions about your documents.\n\nNote: Previous messages in this chat remain visible but are no longer used for context.'
            }
        )
        
    except Exception as e:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), ❌ Error clearing chat: {str(e)}'
            }
        )

def handle_process_command(bot, groupId, creatorId, user_session):
    """Handle document processing"""
    if not user_session['authenticated']:
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), please authenticate first by typing "!auth"'
            }
        )
        return
    
    try:
        user_session['auth_step'] = 'processing'
        
        # Process documents with status updates
        process_documents_with_simple_status(
            bot, groupId, creatorId, user_session['credentials'], user_session['user_email']
        )
        
        user_session['processed_documents'] = True
        user_session['auth_step'] = 'ready'
        
        # Save updated state after processing
        save_auth_state(creatorId, user_session)
        
        # Initialize components after processing documents
        if not rag_bot_state.initialized:
            rag_bot_state.initialize_components()
        
        # Get document counts for final message
        pdf_count = get_document_count(OPENSEARCH_INDEX)
        sheets_count = get_document_count(SHEETS_INDEX)
        
        # Document processing is now handled by the status updater function
        # The final success message is sent automatically
        
    except Exception as e:
        user_session['auth_step'] = 'authenticated'
        bot.sendMessage(
            groupId,
            {
                'text': f'![:Person]({creatorId}), ❌ Error processing documents: {str(e)}'
            }
        )

def handle_status_command(bot, groupId, creatorId, user_session):
    """Handle status check"""
    status_text = f"**Status for {creatorId}:**\n\n"
    
    # Authentication status
    if user_session['authenticated']:
        status_text += f"✅ **Authenticated:** {user_session.get('user_email', 'Unknown')}\n"
    else:
        status_text += "❌ **Not authenticated**\n"
    
    # Processing status
    if user_session['processed_documents']:
        status_text += "✅ **Documents processed**\n"
    else:
        status_text += "❌ **Documents not processed**\n"
    
    # Document counts
    try:
        pdf_count = get_document_count(OPENSEARCH_INDEX)
        sheets_count = get_document_count(SHEETS_INDEX)
        status_text += f"📊 **Document chunks:** {pdf_count} PDFs, {sheets_count} Sheets\n"
    except:
        status_text += "📊 **Document chunks:** Unable to retrieve\n"
    
    # Bot status
    if rag_bot_state.initialized and rag_bot_state.unified_retriever and rag_bot_state.llm:
        status_text += "✅ **Bot ready**\n"
    else:
        status_text += "❌ **Bot not ready**\n"
    
    bot.sendMessage(
        groupId,
        {
            'text': f'![:Person]({creatorId}), {status_text}'
        }
    )

# Additional utility functions for bot management

def get_bot_status() -> Dict[str, Any]:
    """Get the current status of the RAG bot"""
    return {
        'initialized': rag_bot_state.initialized,
        'has_retriever': rag_bot_state.unified_retriever is not None,
        'has_llm': rag_bot_state.llm is not None,
        'user_sessions_count': len(rag_bot_state.user_sessions),
        'document_counts': {
            'pdf_docs': get_document_count(OPENSEARCH_INDEX),
            'sheets': get_document_count(SHEETS_INDEX)
        }
    }

def clear_user_session(user_id: str):
    """Clear a user's session data"""
    if user_id in rag_bot_state.user_sessions:
        del rag_bot_state.user_sessions[user_id]

def get_user_session_info(user_id: str) -> Dict[str, Any]:
    """Get information about a user's session"""
    user_session = rag_bot_state.get_user_session(user_id)
    return {
        'authenticated': user_session['authenticated'],
        'user_email': user_session['user_email'],
        'processed_documents': user_session['processed_documents'],
        'chat_history_length': len(user_session['chat_history']),
        'auth_step': user_session['auth_step']
    } 
