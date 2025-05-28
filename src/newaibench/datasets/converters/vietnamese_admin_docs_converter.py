"""
Vietnamese Administrative Documents dataset converter for NewAIBench.

This converter handles Vietnamese administrative documents that we create
as a sample dataset for demonstrating document image retrieval capabilities
with Vietnamese text.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from .base_converter import BaseDatasetConverter, ConversionConfig

logger = logging.getLogger(__name__)


class VietnameseAdminDocsConverter(BaseDatasetConverter):
    """
    Converter for Vietnamese Administrative Documents dataset.
    
    This dataset contains Vietnamese administrative documents (công văn,
    thông báo, báo cáo, etc.) with OCR text and manually created queries
    for document retrieval evaluation.
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.dataset_name = "vietnamese_admin_docs"
        
    def detect_format(self) -> bool:
        """
        Detect if the input directory contains Vietnamese Admin Docs data.
        
        Returns:
            bool: True if Vietnamese Admin Docs format is detected
        """
        input_path = Path(self.config.input_path)
        
        # Check for specific file patterns
        patterns = [
            "**/vietnamese_admin_docs.json",
            "**/admin_docs_metadata.json",
            "**/cong_van*.json",
            "**/thong_bao*.json",
            "**/vn_docs*.json"
        ]
        
        for pattern in patterns:
            if list(input_path.glob(pattern)):
                logger.info(f"Detected Vietnamese Admin Docs format with pattern: {pattern}")
                return True
        
        # Check for images with Vietnamese document naming patterns
        vn_image_patterns = [
            "**/cong_van_*.jpg", "**/cong_van_*.png",
            "**/thong_bao_*.jpg", "**/thong_bao_*.png",
            "**/bao_cao_*.jpg", "**/bao_cao_*.png",
            "**/bien_ban_*.jpg", "**/bien_ban_*.png",
            "**/giay_chung_nhan_*.jpg", "**/giay_chung_nhan_*.png"
        ]
        
        for pattern in vn_image_patterns:
            if list(input_path.glob(pattern)):
                logger.info(f"Detected Vietnamese document images with pattern: {pattern}")
                return True
                
        # Check for Vietnamese admin docs structure in JSON files
        for json_file in input_path.rglob("*.json"):
            if self._check_vietnamese_docs_structure(json_file):
                return True
                
        return False
    
    def _check_vietnamese_docs_structure(self, json_file: Path) -> bool:
        """Check if JSON file contains Vietnamese admin docs structure."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check for Vietnamese document indicators
            if isinstance(data, dict):
                # Check metadata
                if "vietnamese_admin_docs" in str(data):
                    return True
                    
                # Check for Vietnamese document types
                vn_doc_types = ["cong_van", "thong_bao", "bao_cao", "bien_ban", "giay_chung_nhan"]
                for doc_type in vn_doc_types:
                    if doc_type in str(data):
                        return True
                
                # Check for documents list
                if "documents" in data or "corpus" in data:
                    docs = data.get("documents", data.get("corpus", []))
                    if isinstance(docs, list) and len(docs) > 0:
                        sample_doc = docs[0]
                        if isinstance(sample_doc, dict):
                            # Check for Vietnamese text or document type
                            text_content = str(sample_doc.get("text", "")) + str(sample_doc.get("ocr_text", ""))
                            if any(vn_word in text_content.lower() for vn_word in 
                                  ["công văn", "thông báo", "báo cáo", "biên bản", "chứng nhận", 
                                   "ủy ban", "sở", "phòng", "ban", "tỉnh", "thành phố"]):
                                return True
                                
            elif isinstance(data, list) and len(data) > 0:
                sample = data[0]
                if isinstance(sample, dict):
                    text_content = str(sample.get("text", "")) + str(sample.get("ocr_text", ""))
                    if any(vn_word in text_content.lower() for vn_word in 
                          ["công văn", "thông báo", "báo cáo", "biên bản", "chứng nhận"]):
                        return True
                        
        except Exception:
            pass
            
        return False
    
    def convert_corpus(self) -> Path:
        """
        Convert Vietnamese Admin Docs to NewAIBench corpus format.
        
        Returns:
            Path: Path to the converted corpus file
        """
        logger.info("Converting Vietnamese Administrative Documents corpus...")
        
        corpus_file = self.config.output_path / "corpus.jsonl"
        input_path = Path(self.config.input_path)
        
        seen_docs = set()
        doc_count = 0
        
        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            
            # Process JSON files with document data
            for json_file in input_path.rglob("*.json"):
                if json_file.name in ["queries.json", "qrels.json"]:
                    continue
                    
                logger.info(f"Processing file: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different data structures
                    if isinstance(data, dict):
                        if "documents" in data:
                            docs_data = data["documents"]
                        elif "corpus" in data:
                            docs_data = data["corpus"]
                        else:
                            # Single document
                            docs_data = [data]
                    else:
                        docs_data = data
                    
                    # Process each document
                    for doc_data in docs_data:
                        if not isinstance(doc_data, dict):
                            continue
                            
                        doc_entry = self._process_vietnamese_document(doc_data, json_file.stem)
                        if doc_entry and doc_entry["_id"] not in seen_docs:
                            out_f.write(json.dumps(doc_entry, ensure_ascii=False) + '\n')
                            seen_docs.add(doc_entry["_id"])
                            doc_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing {json_file}: {e}")
                    continue
        
        logger.info(f"Converted {doc_count} Vietnamese documents to corpus")
        self._update_stats("corpus_documents", doc_count)
        
        return corpus_file
    
    def _process_vietnamese_document(self, doc_data: Dict, file_stem: str) -> Optional[Dict]:
        """Process a Vietnamese administrative document."""
        try:
            # Extract document ID
            doc_id = (doc_data.get("doc_id") or 
                     doc_data.get("id") or 
                     doc_data.get("_id") or 
                     f"vn_admin_{file_stem}_{len(doc_data)}")
            
            # Extract text content
            text_content = (doc_data.get("text") or 
                           doc_data.get("ocr_text") or 
                           doc_data.get("content") or "")
            
            # Extract title
            title = (doc_data.get("title") or 
                    doc_data.get("tieu_de") or 
                    doc_data.get("subject") or 
                    f"Tài liệu {doc_id}")
            
            # Extract image path
            image_path = (doc_data.get("image_path") or 
                         doc_data.get("file_path") or 
                         doc_data.get("image") or 
                         f"images/{doc_id}.jpg")
            
            # Determine document type
            doc_type = self._determine_vietnamese_doc_type(text_content, title, doc_data)
            
            # Extract metadata
            metadata = {
                "source": "vietnamese_admin_docs",
                "document_type": doc_type,
                "language": "vietnamese",
                "original_file": file_stem
            }
            
            # Add additional metadata if available
            for key in ["date", "ngay_ban_hanh", "so_van_ban", "noi_ban_hanh", 
                       "loai_van_ban", "do_khan", "do_mat"]:
                if key in doc_data:
                    metadata[key] = doc_data[key]
            
            # Create corpus entry
            corpus_entry = {
                "_id": str(doc_id),
                "title": title,
                "text": text_content,
                "image_path": image_path,
                "metadata": metadata
            }
            
            return corpus_entry
            
        except Exception as e:
            logger.warning(f"Error processing Vietnamese document: {e}")
            return None
    
    def _determine_vietnamese_doc_type(self, text: str, title: str, doc_data: Dict) -> str:
        """Determine the type of Vietnamese administrative document."""
        full_text = (text + " " + title).lower()
        
        # Check explicit document type field
        if "document_type" in doc_data:
            return doc_data["document_type"]
        
        if "loai_van_ban" in doc_data:
            return doc_data["loai_van_ban"]
        
        # Infer from content
        if any(keyword in full_text for keyword in ["công văn", "cong van"]):
            return "cong_van"
        elif any(keyword in full_text for keyword in ["thông báo", "thong bao"]):
            return "thong_bao"
        elif any(keyword in full_text for keyword in ["báo cáo", "bao cao"]):
            return "bao_cao"
        elif any(keyword in full_text for keyword in ["biên bản", "bien ban"]):
            return "bien_ban"
        elif any(keyword in full_text for keyword in ["chứng nhận", "chung nhan", "giấy chứng nhận"]):
            return "giay_chung_nhan"
        elif any(keyword in full_text for keyword in ["quyết định", "quyet dinh"]):
            return "quyet_dinh"
        elif any(keyword in full_text for keyword in ["nghị quyết", "nghi quyet"]):
            return "nghi_quyet"
        else:
            return "van_ban_hanh_chinh"
    
    def convert_queries(self) -> Path:
        """
        Convert Vietnamese queries to NewAIBench queries format.
        
        Returns:
            Path: Path to the converted queries file
        """
        logger.info("Converting Vietnamese queries...")
        
        queries_file = self.config.output_path / "queries.jsonl"
        input_path = Path(self.config.input_path)
        
        query_count = 0
        
        with open(queries_file, 'w', encoding='utf-8') as out_f:
            
            # Look for queries file
            queries_sources = [
                input_path / "queries.json",
                input_path / "queries.jsonl",
                input_path / "cau_hoi.json",
                input_path / "vietnamese_queries.json"
            ]
            
            for queries_path in queries_sources:
                if not queries_path.exists():
                    continue
                    
                logger.info(f"Processing queries from: {queries_path}")
                
                try:
                    if queries_path.suffix == ".jsonl":
                        with open(queries_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    query_data = json.loads(line)
                                    query_entry = self._process_vietnamese_query(query_data)
                                    if query_entry:
                                        out_f.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
                                        query_count += 1
                    else:
                        with open(queries_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict) and "queries" in data:
                            data = data["queries"]
                        
                        if isinstance(data, list):
                            for query_data in data:
                                query_entry = self._process_vietnamese_query(query_data)
                                if query_entry:
                                    out_f.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
                                    query_count += 1
                        elif isinstance(data, dict):
                            # Single query or query dict
                            query_entry = self._process_vietnamese_query(data)
                            if query_entry:
                                out_f.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
                                query_count += 1
                                
                except Exception as e:
                    logger.warning(f"Error processing queries from {queries_path}: {e}")
                    continue
        
        # If no queries file found, generate some sample queries
        if query_count == 0:
            logger.info("No queries file found, generating sample Vietnamese queries...")
            sample_queries = self._generate_sample_vietnamese_queries()
            for query_entry in sample_queries:
                out_f.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
                query_count += 1
        
        logger.info(f"Converted {query_count} Vietnamese queries")
        self._update_stats("queries", query_count)
        
        return queries_file
    
    def _process_vietnamese_query(self, query_data: Dict) -> Optional[Dict]:
        """Process a Vietnamese query entry."""
        try:
            query_id = (query_data.get("query_id") or 
                       query_data.get("id") or 
                       query_data.get("_id"))
            
            query_text = (query_data.get("text") or 
                         query_data.get("query") or 
                         query_data.get("cau_hoi") or 
                         query_data.get("question"))
            
            if not query_id or not query_text:
                return None
            
            return {
                "_id": str(query_id),
                "text": query_text.strip(),
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "query_type": query_data.get("query_type", "search"),
                    "domain": query_data.get("domain", "administrative")
                }
            }
            
        except Exception as e:
            logger.warning(f"Error processing Vietnamese query: {e}")
            return None
    
    def _generate_sample_vietnamese_queries(self) -> List[Dict]:
        """Generate sample Vietnamese queries for testing."""
        sample_queries = [
            {
                "_id": "vn_q_001",
                "text": "Tìm công văn về chính sách mới",
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "query_type": "search",
                    "domain": "administrative"
                }
            },
            {
                "_id": "vn_q_002", 
                "text": "Thông báo về cuộc họp hành chính",
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "query_type": "search",
                    "domain": "administrative"
                }
            },
            {
                "_id": "vn_q_003",
                "text": "Báo cáo kết quả hoạt động năm 2024",
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "query_type": "search",
                    "domain": "administrative"
                }
            },
            {
                "_id": "vn_q_004",
                "text": "Giấy chứng nhận đăng ký kinh doanh",
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "query_type": "search",
                    "domain": "administrative"
                }
            },
            {
                "_id": "vn_q_005",
                "text": "Biên bản họp ủy ban nhân dân",
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "query_type": "search",
                    "domain": "administrative"
                }
            }
        ]
        
        return sample_queries
    
    def convert_qrels(self) -> Path:
        """
        Convert Vietnamese qrels to NewAIBench qrels format.
        
        Returns:
            Path: Path to the converted qrels file
        """
        logger.info("Converting Vietnamese qrels...")
        
        qrels_file = self.config.output_path / "qrels.jsonl"
        input_path = Path(self.config.input_path)
        
        qrel_count = 0
        
        with open(qrels_file, 'w', encoding='utf-8') as out_f:
            
            # Look for qrels file
            qrels_sources = [
                input_path / "qrels.json",
                input_path / "qrels.jsonl",
                input_path / "relevance.json",
                input_path / "vietnamese_qrels.json"
            ]
            
            for qrels_path in qrels_sources:
                if not qrels_path.exists():
                    continue
                    
                logger.info(f"Processing qrels from: {qrels_path}")
                
                try:
                    if qrels_path.suffix == ".jsonl":
                        with open(qrels_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    qrel_data = json.loads(line)
                                    qrel_entry = self._process_vietnamese_qrel(qrel_data)
                                    if qrel_entry:
                                        out_f.write(json.dumps(qrel_entry, ensure_ascii=False) + '\n')
                                        qrel_count += 1
                    else:
                        with open(qrels_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict) and "qrels" in data:
                            data = data["qrels"]
                        
                        if isinstance(data, list):
                            for qrel_data in data:
                                qrel_entry = self._process_vietnamese_qrel(qrel_data)
                                if qrel_entry:
                                    out_f.write(json.dumps(qrel_entry, ensure_ascii=False) + '\n')
                                    qrel_count += 1
                                    
                except Exception as e:
                    logger.warning(f"Error processing qrels from {qrels_path}: {e}")
                    continue
        
        # If no qrels found, generate default relevance for sample queries
        if qrel_count == 0:
            logger.info("No qrels file found, generating default relevance judgments...")
            default_qrels = self._generate_default_vietnamese_qrels()
            for qrel_entry in default_qrels:
                out_f.write(json.dumps(qrel_entry, ensure_ascii=False) + '\n')
                qrel_count += 1
        
        logger.info(f"Converted {qrel_count} Vietnamese qrels")
        self._update_stats("qrels", qrel_count)
        
        return qrels_file
    
    def _process_vietnamese_qrel(self, qrel_data: Dict) -> Optional[Dict]:
        """Process a Vietnamese qrel entry."""
        try:
            query_id = qrel_data.get("query_id")
            doc_id = qrel_data.get("doc_id")
            relevance = qrel_data.get("relevance", qrel_data.get("score", 1))
            
            if not query_id or not doc_id:
                return None
            
            return {
                "query_id": str(query_id),
                "doc_id": str(doc_id),
                "relevance": int(relevance),
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "annotation_method": qrel_data.get("annotation_method", "manual")
                }
            }
            
        except Exception as e:
            logger.warning(f"Error processing Vietnamese qrel: {e}")
            return None
    
    def _generate_default_vietnamese_qrels(self) -> List[Dict]:
        """Generate default qrels for sample queries."""
        # This is a placeholder - in real implementation, these would be manually annotated
        default_qrels = [
            {
                "query_id": "vn_q_001",
                "doc_id": "vn_admin_doc_001",
                "relevance": 2,
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "annotation_method": "auto_generated"
                }
            },
            {
                "query_id": "vn_q_002",
                "doc_id": "vn_admin_doc_002", 
                "relevance": 2,
                "metadata": {
                    "source": "vietnamese_admin_docs",
                    "language": "vietnamese",
                    "annotation_method": "auto_generated"
                }
            }
        ]
        
        return default_qrels
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata information for the converted dataset."""
        base_metadata = super().get_metadata()
        
        base_metadata.update({
            "dataset_name": "vietnamese_admin_docs",
            "description": "Vietnamese Administrative Documents dataset for document image retrieval",
            "original_format": "JSON with Vietnamese document images and OCR text",
            "task_type": "document_image_retrieval",
            "domain": "Vietnamese administrative documents",
            "language": "Vietnamese",
            "document_types": ["cong_van", "thong_bao", "bao_cao", "bien_ban", "giay_chung_nhan", "quyet_dinh", "nghi_quyet"],
            "ocr_requirements": "Vietnamese text recognition capability",
            "relevance_scale": "0-2 (0: not relevant, 1: relevant, 2: highly relevant)",
            "annotation_method": "Manual annotation by Vietnamese speakers",
            "use_cases": ["Government document search", "Administrative document management", "Vietnamese OCR evaluation"]
        })
        
        return base_metadata
