"""
spaCy-based Metadata Handler for OpenSearch

This module provides spaCy-powered metadata classification and query handling
for the document management system.
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.matcher import Matcher
from opensearchpy import OpenSearch

# Configuration
OPENSEARCH_URL = os.getenv('OPENSEARCH_URL', 'http://localhost:9200')
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME', '')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '')
METADATA_INDEX = "file_metadata"

@dataclass
class QueryClassification:
    """Result of query classification"""
    classification: str  # 'content' or 'metadata'
    confidence: float
    reasoning: str
    query_type: Optional[str] = None  # 'temporal', 'relationship', 'property', etc.

@dataclass
class MetadataQuery:
    """Structured metadata query"""
    query_type: str
    parameters: Dict[str, Any]
    original_query: str

class SpacyMetadataHandler:
    """spaCy-based metadata classification and handling"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy model: en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize OpenSearch client
        self.opensearch_client = self._create_opensearch_client()
        
        # Setup patterns and matcher
        self._setup_patterns()
    
    def _create_opensearch_client(self) -> Optional[OpenSearch]:
        """Create OpenSearch client connection"""
        try:
            if not OPENSEARCH_USERNAME and not OPENSEARCH_PASSWORD:
                client = OpenSearch(
                    hosts=[OPENSEARCH_URL],
                    use_ssl=False,
                    verify_certs=False,
                    ssl_show_warn=False
                )
            else:
                client = OpenSearch(
                    hosts=[OPENSEARCH_URL],
                    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
                    use_ssl=OPENSEARCH_URL.startswith('https'),
                    verify_certs=False,
                    ssl_show_warn=False
                )
            return client
        except Exception as e:
            print(f"Failed to connect to OpenSearch: {str(e)}")
            return None
    
    def _setup_patterns(self):
        """Setup spaCy patterns for query classification"""
        self.matcher = Matcher(self.nlp.vocab)
        
        # Metadata patterns
        metadata_patterns = [
            # Temporal patterns
            [{"LOWER": {"IN": ["last", "recent", "yesterday", "today", "this"]}}],
            [{"LOWER": "files"}, {"LOWER": "from"}],
            [{"LOWER": "modified"}, {"LOWER": "when"}],
            [{"LOWER": "created"}, {"LOWER": "when"}],
            [{"LOWER": "opened"}, {"LOWER": "when"}],
            [{"LOWER": "viewed"}, {"LOWER": "when"}],
            
            # Relationship patterns
            [{"LOWER": "who"}, {"LOWER": {"IN": ["shared", "owns", "created", "has"]}}],
            [{"LOWER": "shared"}, {"LOWER": "by"}],
            [{"LOWER": "owned"}, {"LOWER": "by"}],
            [{"LOWER": "permissions"}],
            [{"LOWER": "access"}],
            
            # Property patterns
            [{"LOWER": {"IN": ["largest", "smallest", "biggest"]}}],
            [{"LOWER": "how"}, {"LOWER": "many"}],
            [{"LOWER": "count"}, {"LOWER": "of"}],
            [{"LOWER": "file"}, {"LOWER": "type"}],
            [{"LOWER": "file"}, {"LOWER": "size"}],
            [{"LOWER": "number"}, {"LOWER": "of"}],
            
            # General metadata patterns
            [{"LOWER": "show"}, {"LOWER": "all"}],
            [{"LOWER": "list"}, {"LOWER": "files"}],
            [{"LOWER": "metadata"}],
            [{"LOWER": "properties"}],
            [{"LOWER": "structure"}],
            [{"LOWER": "organization"}],
        ]
        
        # Content patterns
        content_patterns = [
            # Content patterns
            [{"LOWER": "what"}, {"LOWER": "does"}],
            [{"LOWER": "what"}, {"LOWER": "is"}],
            [{"LOWER": "what"}, {"LOWER": "are"}],
            [{"LOWER": "explain"}],
            [{"LOWER": "summarize"}],
            [{"LOWER": "find"}, {"LOWER": "information"}],
            [{"LOWER": "find"}, {"LOWER": "file"}, {"LOWER": "containing"}],
            [{"LOWER": "find"}, {"LOWER": "containing"}],
            [{"LOWER": "find"}, {"LOWER": "file"}, {"LOWER": "has"}],
            [{"LOWER": "find"}, {"LOWER": "file"}, {"LOWER": "includes"}],
            [{"LOWER": "find"}, {"LOWER": "file"}, {"LOWER": "shows"}],
            [{"LOWER": "which"}, {"LOWER": "file"}, {"LOWER": "has"}],
            [{"LOWER": "which"}, {"LOWER": "file"}, {"LOWER": "contains"}],
            [{"LOWER": "which"}, {"LOWER": "file"}, {"LOWER": "includes"}],
            [{"LOWER": "key"}, {"LOWER": "points"}],
            [{"LOWER": "topics"}],
            [{"LOWER": "content"}],
            [{"LOWER": "says"}, {"LOWER": "about"}],
            [{"LOWER": "discusses"}],
            [{"LOWER": "mentions"}],
            [{"LOWER": "analyze"}],
            [{"LOWER": "describe"}],
            [{"LOWER": "details"}],
            [{"LOWER": "facts"}],
            [{"LOWER": "data"}],
            [{"LOWER": "reviews"}],
            [{"LOWER": "products"}],
            [{"LOWER": "companies"}],
        ]
        
        # Add patterns to matcher
        for pattern in metadata_patterns:
            self.matcher.add("METADATA", [pattern])
        for pattern in content_patterns:
            self.matcher.add("CONTENT", [pattern])
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query as either 'content' or 'metadata' using spaCy
        """
        try:
            doc = self.nlp(query.lower())
            
            # Get matches
            matches = self.matcher(doc)
            
            # Count matches by type
            metadata_score = sum(1 for match_id, start, end in matches 
                               if self.nlp.vocab.strings[match_id] == "METADATA")
            content_score = sum(1 for match_id, start, end in matches 
                              if self.nlp.vocab.strings[match_id] == "CONTENT")
            
            # Additional keyword scoring
            metadata_keywords = [
                "size", "date", "time", "when", "who", "shared", "owned", "created", 
                "modified", "opened", "viewed", "count", "number", "type", "format", 
                "properties", "metadata", "structure", "organization", "permissions", "access"
            ]
            
            content_keywords = [
                "what", "explain", "information", "content", "topic", "topics",
                "summarize", "analyze", "describe", "details", "facts", "data",
                "says", "discusses", "mentions", "key", "points", "conclusion",
                "method", "process", "steps", "procedure", "how", "why",
                "containing", "with", "about", "reviews", "products", "companies"
            ]
            
            # Score based on keyword presence
            for token in doc:
                if token.text in metadata_keywords:
                    metadata_score += 0.3
                if token.text in content_keywords:
                    content_score += 0.3
            
            # Special handling for content-seeking queries (these are content queries)
            content_seeking_words = ["containing", "has", "includes", "shows", "with", "about"]
            if any(word in doc.text for word in content_seeking_words):
                content_score += 2.0  # Strong boost for content queries
            
            # Determine classification
            if metadata_score > content_score:
                classification = "metadata"
                confidence = min(metadata_score / (metadata_score + content_score + 1), 0.95)
                reasoning = f"spaCy pattern matching: metadata score {metadata_score:.1f}, content score {content_score:.1f}"
                query_type = self._classify_metadata_query_type(doc)
            else:
                classification = "content"
                confidence = min(content_score / (metadata_score + content_score + 1), 0.95)
                reasoning = f"spaCy pattern matching: content score {content_score:.1f}, metadata score {metadata_score:.1f}"
                query_type = None
            
            return QueryClassification(
                classification=classification,
                confidence=confidence,
                reasoning=reasoning,
                query_type=query_type
            )
            
        except Exception as e:
            print(f"Error in spaCy query classification: {str(e)}")
            return self._fallback_classification(query)
    
    def _classify_metadata_query_type(self, doc: Doc) -> str:
        """Classify the specific type of metadata query"""
        text = doc.text.lower()
        
        # Temporal queries
        temporal_keywords = [
            "last week", "yesterday", "today", "recent", "recently", 
            "this month", "this week", "modified", "opened", "viewed", 
            "created", "when", "date", "time"
        ]
        if any(keyword in text for keyword in temporal_keywords):
            return "temporal"
        
        # Relationship queries
        relationship_keywords = [
            "shared by", "owned by", "created by", "who", "person", 
            "user", "team", "permissions", "access"
        ]
        if any(keyword in text for keyword in relationship_keywords):
            return "relationship"
        
        # Property queries
        property_keywords = [
            "largest", "smallest", "type", "format", "size", "biggest", 
            "how many", "count", "number of", "file type"
        ]
        if any(keyword in text for keyword in property_keywords):
            return "property"
        
        # Content search queries
        content_keywords = [
            "contains", "about", "topic", "subject", "find", "search"
        ]
        if any(keyword in text for keyword in content_keywords):
            return "content_search"
        
        return "general"
    
    def _fallback_classification(self, query: str) -> QueryClassification:
        """Fallback classification using simple keyword matching"""
        query_lower = query.lower()
        
        metadata_keywords = [
            'how many', 'count', 'number of', 'files', 'documents',
            'file type', 'file size', 'file name', 'file format',
            'when', 'date', 'time', 'modified', 'created', 'processed',
            'recent', 'oldest', 'newest', 'largest', 'smallest',
            'show me all', 'list', 'what files', 'which files',
            'file properties', 'metadata', 'structure', 'organization'
        ]
        
        content_keywords = [
            'what does', 'what is', 'what are', 'find', 'search',
            'information about', 'data about', 'content', 'says',
            'explain', 'describe', 'analyze', 'summarize', 'key points',
            'topics', 'subjects', 'details', 'facts', 'information'
        ]
        
        metadata_score = sum(1 for keyword in metadata_keywords if keyword in query_lower)
        content_score = sum(1 for keyword in content_keywords if keyword in query_lower)
        
        if metadata_score > content_score:
            return QueryClassification(
                classification="metadata",
                confidence=0.7,
                reasoning="Fallback keyword-based classification: query contains metadata-related terms",
                query_type=self._classify_metadata_query_type(self.nlp(query_lower))
            )
        else:
            return QueryClassification(
                classification="content",
                confidence=0.7,
                reasoning="Fallback keyword-based classification: query appears to ask about content"
            )
    
    def create_metadata_index(self) -> bool:
        """Create the metadata index in OpenSearch"""
        if not self.opensearch_client:
            return False
        
        try:
            # Check if index already exists
            if self.opensearch_client.indices.exists(index=METADATA_INDEX):
                print(f"✅ Metadata index {METADATA_INDEX} already exists")
                return True
            
            # Define the index mapping
            index_mapping = {
                "mappings": {
                    "properties": {
                        # Basic file info
                        "file_id": {"type": "keyword"},
                        "file_name": {"type": "text"},
                        "file_type": {"type": "keyword"},
                        "mime_type": {"type": "keyword"},
                        
                        # Temporal fields
                        "created_time": {"type": "date"},
                        "modified_time": {"type": "date"},
                        "viewed_by_me_time": {"type": "date"},
                        "processed_time": {"type": "date"},
                        
                        # Size and content info
                        "file_size_mb": {"type": "float"},
                        "page_count": {"type": "integer"},
                        "word_count": {"type": "integer"},
                        "sheet_count": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        
                        # Ownership and permissions
                        "owners": {
                            "type": "nested",
                            "properties": {
                                "displayName": {"type": "text"},
                                "emailAddress": {"type": "keyword"},
                                "permissionId": {"type": "keyword"}
                            }
                        },
                        "permissions": {
                            "type": "nested", 
                            "properties": {
                                "displayName": {"type": "text"},
                                "emailAddress": {"type": "keyword"},
                                "role": {"type": "keyword"},
                                "type": {"type": "keyword"},
                                "permissionId": {"type": "keyword"}
                            }
                        },
                        
                        # File status
                        "trashed": {"type": "boolean"},
                        "starred": {"type": "boolean"},
                        "shared": {"type": "boolean"},
                        
                        # Hierarchical structure
                        "parents": {"type": "keyword"},
                        "folder_path": {"type": "text"},
                        
                        # User context
                        "user_email": {"type": "keyword"},
                        "access_level": {"type": "keyword"},
                        
                        # Additional metadata
                        "web_view_link": {"type": "keyword"},
                        "web_content_link": {"type": "keyword"},
                        "thumbnail_link": {"type": "keyword"},
                        "capabilities": {"type": "object"},
                        "export_links": {"type": "object"},
                        "app_properties": {"type": "object"},
                        "properties": {"type": "object"}
                    }
                },
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                }
            }
            
            # Create the index
            self.opensearch_client.indices.create(
                index=METADATA_INDEX,
                body=index_mapping
            )
            
            print(f"✅ Created metadata index: {METADATA_INDEX}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating metadata index: {str(e)}")
            return False
    
    def store_metadata_in_opensearch(self, file_metadata: Dict, user_email: str) -> bool:
        """Store metadata in OpenSearch"""
        if not self.opensearch_client:
            return False
        
        try:
            # Ensure index exists
            if not self.opensearch_client.indices.exists(index=METADATA_INDEX):
                self.create_metadata_index()
            
            success_count = 0
            for file_hash, metadata in file_metadata.items():
                try:
                    # Prepare document for OpenSearch
                    document = {
                        "user_email": user_email,
                        "file_id": metadata.get('file_id', ''),
                        "file_name": metadata.get('file_name', ''),
                        "file_type": metadata.get('file_type', ''),
                        "mime_type": metadata.get('mime_type', ''),
                        "file_size_mb": metadata.get('file_size_mb', 0.0),
                        "created_time": metadata.get('created_time', ''),
                        "modified_time": metadata.get('modified_time', ''),
                        "viewed_by_me_time": metadata.get('viewed_by_me_time', ''),
                        "processed_time": metadata.get('processed_time', ''),
                        "page_count": metadata.get('page_count', 0),
                        "word_count": metadata.get('word_count', 0),
                        "sheet_count": metadata.get('sheet_count', 0),
                        "total_chunks": metadata.get('total_chunks', 0),
                        "trashed": metadata.get('trashed', False),
                        "starred": metadata.get('starred', False),
                        "shared": metadata.get('shared', False),
                        "parents": metadata.get('parents', []),
                        "web_view_link": metadata.get('web_view_link', ''),
                        "web_content_link": metadata.get('web_content_link', ''),
                        "thumbnail_link": metadata.get('thumbnail_link', ''),
                        "capabilities": metadata.get('capabilities', {}),
                        "export_links": metadata.get('export_links', {}),
                        "app_properties": metadata.get('app_properties', {}),
                        "properties": metadata.get('properties', {}),
                        
                        # Nested objects
                        "owners": metadata.get('owners', []),
                        "permissions": metadata.get('permissions', [])
                    }
                    
                    # Index the document
                    self.opensearch_client.index(
                        index=METADATA_INDEX,
                        id=file_hash,
                        body=document
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error storing metadata for file {file_hash}: {str(e)}")
                    continue
            
            print(f"✅ Successfully stored {success_count} metadata documents in OpenSearch")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error storing metadata in OpenSearch: {str(e)}")
            return False
    
    def handle_metadata_query(self, query: str, user_email: str) -> str:
        """Handle metadata queries using spaCy and OpenSearch"""
        try:
            # Classify the query type
            classification = self.classify_query(query)
            
            if classification.classification != "metadata":
                return "This query appears to be about content, not metadata."
            
            # Route to appropriate handler based on query type
            if classification.query_type == "temporal":
                return self._handle_temporal_query(query, user_email)
            elif classification.query_type == "relationship":
                return self._handle_relationship_query(query, user_email)
            elif classification.query_type == "property":
                return self._handle_property_query(query, user_email)
            else:
                return self._handle_general_metadata_query(query, user_email)
                
        except Exception as e:
            print(f"Error handling metadata query: {str(e)}")
            return f"❌ Error processing metadata query: {str(e)}"
    
    def _handle_temporal_query(self, query: str, user_email: str) -> str:
        """Handle temporal queries (time-based)"""
        try:
            # Parse temporal intent
            temporal_info = self._extract_temporal_intent(query)
            
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_email": user_email}}
                        ]
                    }
                },
                "sort": [{"modified_time": {"order": "desc"}}],
                "size": 20
            }
            
            # Add temporal filters
            if temporal_info.get("field") and temporal_info.get("range"):
                search_body["query"]["bool"]["must"].append({
                    "range": {
                        temporal_info["field"]: temporal_info["range"]
                    }
                })
            
            # Execute search
            results = self.opensearch_client.search(index=METADATA_INDEX, body=search_body)
            
            # Format results
            return self._format_temporal_results(results, temporal_info)
            
        except Exception as e:
            print(f"Error handling temporal query: {str(e)}")
            return f"❌ Error processing temporal query: {str(e)}"
    
    def _handle_relationship_query(self, query: str, user_email: str) -> str:
        """Handle relationship queries (people-based)"""
        try:
            # Extract person name
            person_name = self._extract_person_name(query)
            
            # If no person name found, try to find file name and show its permissions
            if not person_name:
                # Look for file name in the query
                doc = self.nlp(query)
                file_name = None
                
                # Look for quoted text or specific file names
                for token in doc:
                    if token.text.lower() in ["file", "document", "pdf", "sheet", "doc"]:
                        # Look for the next token as potential file name
                        if token.i + 1 < len(doc):
                            potential_name = doc[token.i + 1].text
                            if potential_name.lower() not in ["is", "was", "has", "the", "a", "an"]:
                                file_name = potential_name
                                break
                
                # If still no file name, try to extract from "the file X" pattern
                if not file_name:
                    query_lower = query.lower()
                    if "the file" in query_lower:
                        parts = query_lower.split("the file")
                        if len(parts) > 1:
                            file_name = parts[1].strip().split()[0]  # Take first word after "the file"
                
                if file_name:
                    # Search for the specific file and show its permissions
                    search_body = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"user_email": user_email}},
                                    {"match": {"file_name": file_name}}
                                ]
                            }
                        },
                        "size": 5
                    }
                    
                    results = self.opensearch_client.search(index=METADATA_INDEX, body=search_body)
                    hits = results.get('hits', {}).get('hits', [])
                    
                    if hits:
                        source = hits[0]['_source']
                        file_name = source.get('file_name', 'Unknown')
                        permissions = source.get('permissions', [])
                        owners = source.get('owners', [])
                        web_view_link = source.get('web_view_link', '')
                        
                        response = f"**Permissions for '{file_name}':**\n\n"
                        
                        # Show owners
                        if owners:
                            response += "**Owners:**\n"
                            for owner in owners:
                                response += f"   • {owner.get('displayName', 'Unknown')} ({owner.get('emailAddress', 'Unknown')})\n"
                            response += "\n"
                        
                        # Show permissions
                        if permissions:
                            response += "**Permissions:**\n"
                            for perm in permissions:
                                role = perm.get('role', 'Unknown')
                                email = perm.get('emailAddress', 'Unknown')
                                name = perm.get('displayName', email)
                                response += f"   • {name} ({email}) - {role}\n"
                        else:
                            response += "**Permissions:** No specific permissions found.\n"
                        
                        if web_view_link:
                            response += f"\n**Link:** {web_view_link}\n"
                        
                        return response
                    else:
                        return f"File '{file_name}' not found in your documents."
            
            # Original person-based search
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_email": user_email}},
                            {
                                "nested": {
                                    "path": "permissions",
                                    "query": {
                                        "bool": {
                                            "should": [
                                                {"match": {"permissions.displayName": person_name}},
                                                {"match": {"permissions.emailAddress": person_name}}
                                            ]
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [{"modified_time": {"order": "desc"}}],
                "size": 20
            }
            
            # Execute search
            results = self.opensearch_client.search(index=METADATA_INDEX, body=search_body)
            
            # Format results
            return self._format_relationship_results(results, person_name)
            
        except Exception as e:
            print(f"Error handling relationship query: {str(e)}")
            return f"Error processing relationship query: {str(e)}"
    
    def _handle_property_query(self, query: str, user_email: str) -> str:
        """Handle property queries (file properties)"""
        try:
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_email": user_email}}
                        ]
                    }
                },
                "sort": [{"modified_time": {"order": "desc"}}],
                "size": 20
            }
            
            # Add property-specific filters
            query_lower = query.lower()
            
            if "largest" in query_lower or "biggest" in query_lower:
                search_body["sort"] = [{"file_size_mb": {"order": "desc"}}]
            elif "smallest" in query_lower:
                search_body["sort"] = [{"file_size_mb": {"order": "asc"}}]
            elif "pdf" in query_lower:
                search_body["query"]["bool"]["must"].append({"term": {"mime_type": "application/pdf"}})
            elif "google doc" in query_lower or "document" in query_lower:
                search_body["query"]["bool"]["must"].append({"term": {"mime_type": "application/vnd.google-apps.document"}})
            elif "sheet" in query_lower or "spreadsheet" in query_lower:
                search_body["query"]["bool"]["must"].append({"term": {"mime_type": "application/vnd.google-apps.spreadsheet"}})
            
            # Execute search
            results = self.opensearch_client.search(index=METADATA_INDEX, body=search_body)
            
            # Format results
            return self._format_property_results(results, query)
            
        except Exception as e:
            print(f"Error handling property query: {str(e)}")
            return f"Error processing property query: {str(e)}"
    
    def _handle_general_metadata_query(self, query: str, user_email: str) -> str:
        """Handle general metadata queries using text search"""
        try:
            # Build text search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_email": user_email}},
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["file_name", "file_type", "mime_type"]
                                }
                            }
                        ]
                    }
                },
                "sort": [{"modified_time": {"order": "desc"}}],
                "size": 10
            }
            
            # Execute search
            results = self.opensearch_client.search(index=METADATA_INDEX, body=search_body)
            
            # Format results
            return self._format_general_results(results, query)
            
        except Exception as e:
            print(f"Error handling general metadata query: {str(e)}")
            return f"Error processing general metadata query: {str(e)}"
    
    def _extract_temporal_intent(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from query"""
        query_lower = query.lower()
        
        temporal_info = {
            "field": "modified_time",  # default field
            "range": {}
        }
        
        # Determine time field
        if "opened" in query_lower or "viewed" in query_lower:
            temporal_info["field"] = "viewed_by_me_time"
        elif "created" in query_lower:
            temporal_info["field"] = "created_time"
        elif "modified" in query_lower or "changed" in query_lower:
            temporal_info["field"] = "modified_time"
        
        # Determine time range
        if "last week" in query_lower:
            temporal_info["range"] = {"gte": "now-7d", "lte": "now"}
        elif "yesterday" in query_lower:
            temporal_info["range"] = {"gte": "now-1d", "lte": "now"}
        elif "today" in query_lower:
            temporal_info["range"] = {"gte": "now/d", "lte": "now"}
        elif "this month" in query_lower:
            temporal_info["range"] = {"gte": "now/M", "lte": "now"}
        elif "this week" in query_lower:
            temporal_info["range"] = {"gte": "now/w", "lte": "now"}
        elif "recent" in query_lower or "recently" in query_lower:
            temporal_info["range"] = {"gte": "now-3d", "lte": "now"}
        
        return temporal_info
    
    def _extract_person_name(self, query: str) -> Optional[str]:
        """Extract person name from query using spaCy NER"""
        doc = self.nlp(query)
        
        # Look for person entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        
        # Fallback: look for common patterns
        query_lower = query.lower()
        
        if "shared by" in query_lower:
            return query_lower.split("shared by")[-1].strip()
        elif "owned by" in query_lower:
            return query_lower.split("owned by")[-1].strip()
        elif "created by" in query_lower:
            return query_lower.split("created by")[-1].strip()
        
        return None
    
    def _format_temporal_results(self, results: Dict, temporal_info: Dict) -> str:
        """Format temporal query results"""
        hits = results.get('hits', {}).get('hits', [])
        
        if not hits:
            return "No files found matching your temporal criteria."
        
        response = f"**Files {temporal_info.get('field', 'modified')} in the specified time period:**\n\n"
        
        for i, hit in enumerate(hits[:10], 1):
            source = hit['_source']
            file_name = source.get('file_name', 'Unknown')
            file_type = source.get('file_type', 'Unknown')
            file_size = source.get('file_size_mb', 0)
            modified_time = source.get('modified_time', 'Unknown')
            web_view_link = source.get('web_view_link', '')
            
            response += f"{i}. **{file_name}** ({file_type})\n"
            response += f"   Size: {file_size:.2f} MB\n"
            response += f"   Modified: {modified_time[:10]}\n"
            if web_view_link:
                response += f"   Link: {web_view_link}\n"
            response += "\n"
        
        return response
    
    def _format_relationship_results(self, results: Dict, person_name: str) -> str:
        """Format relationship query results"""
        hits = results.get('hits', {}).get('hits', [])
        
        if not hits:
            return f"No files found shared by or owned by '{person_name}'."
        
        response = f"**Files related to '{person_name}':**\n\n"
        
        for i, hit in enumerate(hits[:10], 1):
            source = hit['_source']
            file_name = source.get('file_name', 'Unknown')
            file_type = source.get('file_type', 'Unknown')
            web_view_link = source.get('web_view_link', '')
            permissions = source.get('permissions', [])
            
            response += f"{i}. **{file_name}** ({file_type})\n"
            
            # Show permission details
            for perm in permissions:
                if person_name.lower() in perm.get('displayName', '').lower() or \
                   person_name.lower() in perm.get('emailAddress', '').lower():
                    role = perm.get('role', 'Unknown')
                    response += f"   Role: {role}\n"
                    break
            
            if web_view_link:
                response += f"   Link: {web_view_link}\n"
            response += "\n"
        
        return response
    
    def _format_property_results(self, results: Dict, query: str) -> str:
        """Format property query results"""
        hits = results.get('hits', {}).get('hits', [])
        
        if not hits:
            return "No files found matching your criteria."
        
        query_lower = query.lower()
        
        if "largest" in query_lower or "biggest" in query_lower:
            response = "**Largest files:**\n\n"
        elif "smallest" in query_lower:
            response = "**Smallest files:**\n\n"
        elif "pdf" in query_lower:
            response = "**PDF files:**\n\n"
        elif "google doc" in query_lower or "document" in query_lower:
            response = "**Google Documents:**\n\n"
        elif "sheet" in query_lower or "spreadsheet" in query_lower:
            response = "**Google Sheets:**\n\n"
        else:
            response = "**Files matching your criteria:**\n\n"
        
        for i, hit in enumerate(hits[:10], 1):
            source = hit['_source']
            file_name = source.get('file_name', 'Unknown')
            file_type = source.get('file_type', 'Unknown')
            file_size = source.get('file_size_mb', 0)
            modified_time = source.get('modified_time', 'Unknown')
            web_view_link = source.get('web_view_link', '')
            
            response += f"{i}. **{file_name}** ({file_type})\n"
            response += f"   Size: {file_size:.2f} MB\n"
            response += f"   Modified: {modified_time[:10]}\n"
            if web_view_link:
                response += f"   Link: {web_view_link}\n"
            response += "\n"
        
        return response
    
    def _format_general_results(self, results: Dict, query: str) -> str:
        """Format general metadata query results"""
        hits = results.get('hits', {}).get('hits', [])
        
        if not hits:
            return "No files found matching your query."
        
        response = f"**Files matching '{query}':**\n\n"
        
        for i, hit in enumerate(hits[:10], 1):
            source = hit['_source']
            file_name = source.get('file_name', 'Unknown')
            file_type = source.get('file_type', 'Unknown')
            file_size = source.get('file_size_mb', 0)
            modified_time = source.get('modified_time', 'Unknown')
            web_view_link = source.get('web_view_link', '')
            
            response += f"{i}. **{file_name}** ({file_type})\n"
            response += f"   Size: {file_size:.2f} MB\n"
            response += f"   Modified: {modified_time[:10]}\n"
            if web_view_link:
                response += f"   Link: {web_view_link}\n"
            response += "\n"
        
        return response

# Global instance
spacy_metadata_handler = None

def get_spacy_metadata_handler() -> SpacyMetadataHandler:
    """Get or create the global spaCy metadata handler instance"""
    global spacy_metadata_handler
    if spacy_metadata_handler is None:
        spacy_metadata_handler = SpacyMetadataHandler()
    return spacy_metadata_handler 