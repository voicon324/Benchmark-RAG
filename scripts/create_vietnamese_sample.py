#!/usr/bin/env python3
"""
Vietnamese Administrative Documents Sample Dataset Creator

Creates a realistic sample dataset of Vietnamese administrative documents
for testing and demonstration purposes.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

# Image generation imports (optional)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class VietnameseSampleCreator:
    """Creates sample Vietnamese administrative documents dataset."""
    
    # Vietnamese document templates
    DOCUMENT_TEMPLATES = {
        "cong_van": {
            "name": "Công văn",
            "template": """
UBND THÀNH PHỐ HÀ NỘI
SỞ GIÁO DỤC VÀ ĐÀO TẠO

Số: {so_van_ban}/SGDĐT-{phong_ban}
V/v: {tieu_de}

Kính gửi: {noi_nhan}

Căn cứ Luật Giáo dục năm 2019;
Căn cứ Nghị định số 81/2021/NĐ-CP ngày 27/8/2021;

Sở Giáo dục và Đào tạo thông báo:

{noi_dung}

Đề nghị Quý cơ quan phối hợp thực hiện.

Trân trọng cảm ơn!

                                    TM. GIÁM ĐỐC
                                    PHÓ GIÁM ĐỐC
                                    
                                    {nguoi_ky}
            """,
            "metadata_template": {
                "loai_van_ban": "công văn",
                "co_quan_ban_hanh": "Sở Giáo dục và Đào tạo Hà Nội",
                "nguoi_ky": ["Nguyễn Văn A", "Trần Thị B", "Lê Văn C"],
                "phong_ban": ["QLCL", "GDTX", "GDMN", "GDPT"],
                "subjects": [
                    "Tổ chức kỳ thi tốt nghiệp THPT",
                    "Triển khai chương trình giáo dục phổ thông mới",
                    "Hướng dẫn tuyển sinh đầu cấp",
                    "Kiểm tra chất lượng giáo dục",
                    "Bồi dưỡng giáo viên"
                ]
            }
        },
        
        "thong_bao": {
            "name": "Thông báo",
            "template": """
UBND PHƯỜNG {phuong}
QUẬN {quan}, TP HÀ NỘI

THÔNG BÁO
V/v: {tieu_de}

Căn cứ Luật Tổ chức chính quyền địa phương;
Căn cứ Quyết định số {so_quyet_dinh} của UBND thành phố;

UBND phường {phuong} thông báo:

{noi_dung}

Mọi thắc mắc xin liên hệ:
- Địa chỉ: {dia_chi}
- Điện thoại: {dien_thoai}

                                    CHỦ TỊCH UBND PHƯỜNG
                                    
                                    {chu_tich}
            """,
            "metadata_template": {
                "loai_van_ban": "thông báo",
                "co_quan_ban_hanh": "UBND Phường",
                "phuong": ["Cầu Giấy", "Đống Đa", "Ba Đình", "Hoàn Kiếm", "Hai Bà Trưng"],
                "quan": ["Cầu Giấy", "Đống Đa", "Ba Đình", "Hoàn Kiếm", "Hai Bà Trưng"],
                "chu_tich": ["Nguyễn Văn An", "Trần Thị Bích", "Lê Văn Cường"],
                "subjects": [
                    "Thông báo lịch cắt điện",
                    "Tạm ngừng cấp nước sinh hoạt",
                    "Tổ chức họp dân phố",
                    "Triển khai dự án cải tạo đường",
                    "Hướng dẫn thủ tục hành chính"
                ]
            }
        },
        
        "bao_cao": {
            "name": "Báo cáo",
            "template": """
UBND THÀNH PHỐ HÀ NỘI
SỞ TÀI NGUYÊN VÀ MÔI TRƯỜNG

BÁO CÁO
Tình hình thực hiện {chu_de} {thoi_gian}

I. TÌNH HÌNH CHUNG

{tinh_hình_chung}

II. KẾT QUẢ ĐẠT ĐƯỢC

{ket_qua}

III. TỒN TẠI, HẠN CHẾ

{ton_tai}

IV. ĐỊNH HƯỚNG THỜI GIAN TỚI

{dinh_huong}

                                    GIÁM ĐỐC SỞ
                                    
                                    {giam_doc}
            """,
            "metadata_template": {
                "loai_van_ban": "báo cáo",
                "co_quan_ban_hanh": "Sở Tài nguyên và Môi trường",
                "giam_doc": ["Phạm Văn D", "Hoàng Thị E", "Đỗ Văn F"],
                "chu_de": [
                    "công tác bảo vệ môi trường",
                    "quản lý đất đai",
                    "khai thác tài nguyên nước",
                    "ứng phó với biến đổi khí hậu"
                ],
                "thoi_gian": ["quý I", "quý II", "quý III", "quý IV", "6 tháng đầu năm", "năm 2024"]
            }
        },
        
        "quyet_dinh": {
            "name": "Quyết định",
            "template": """
UBND THÀNH PHỐ HÀ NỘI

QUYẾT ĐỊNH
Số: {so_quyet_dinh}/QĐ-UBND

V/v: {tieu_de}

CHỦ TỊCH UBND THÀNH PHỐ HÀ NỘI

Căn cứ Luật Tổ chức chính quyền địa phương 2015;
Căn cứ {can_cu_phap_ly};
Theo đề nghị của {de_nghi};

QUYẾT ĐỊNH:

Điều 1. {dieu_1}

Điều 2. {dieu_2}

Điều 3. Quyết định này có hiệu lực từ ngày ký.

                                    CHỦ TỊCH
                                    
                                    {chu_tich}
            """,
            "metadata_template": {
                "loai_van_ban": "quyết định",
                "co_quan_ban_hanh": "UBND Thành phố Hà Nội",
                "chu_tich": ["Nguyễn Đức Chung", "Chu Ngọc Anh", "Trần Sỹ Thanh"],
                "subjects": [
                    "Phê duyệt dự án đầu tư công",
                    "Ban hành quy chế tổ chức",
                    "Điều chỉnh quy hoạch đô thị",
                    "Quy định về thu phí dịch vụ"
                ]
            }
        }
    }
    
    def __init__(self, output_path: str):
        """
        Initialize the Vietnamese sample creator.
        
        Args:
            output_path: Directory to create the sample dataset
        """
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directories
        self.images_dir = self.output_path / "images"
        self.ocr_dir = self.output_path / "ocr"
        
    def create_sample_dataset(self, num_docs: int = 50) -> bool:
        """
        Create a complete sample dataset.
        
        Args:
            num_docs: Number of sample documents to create
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Creating Vietnamese sample dataset with {num_docs} documents...")
            
            # Create directory structure
            self._create_directory_structure()
            
            # Generate documents
            documents = self._generate_documents(num_docs)
            
            # Create images (if PIL available)
            if PIL_AVAILABLE:
                self._create_document_images(documents)
            else:
                self.logger.warning("PIL not available - creating text files only")
                self._create_text_files(documents)
            
            # Create dataset files
            self._create_corpus_file(documents)
            self._create_queries_file(documents)
            self._create_qrels_file(documents)
            self._create_metadata_file(documents)
            
            self.logger.info(f"✓ Sample dataset created successfully at: {self.output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sample dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_directory_structure(self):
        """Create necessary directories."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.ocr_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Created directory structure at: {self.output_path}")
    
    def _generate_documents(self, num_docs: int) -> List[Dict[str, Any]]:
        """Generate document data."""
        documents = []
        doc_types = list(self.DOCUMENT_TEMPLATES.keys())
        
        for i in range(num_docs):
            doc_type = random.choice(doc_types)
            template_info = self.DOCUMENT_TEMPLATES[doc_type]
            
            # Generate document content
            doc_content = self._generate_document_content(doc_type, template_info)
            
            doc = {
                "doc_id": f"vn_admin_{doc_type}_{i+1:03d}",
                "document_type": doc_type,
                "title": doc_content["title"],
                "text": doc_content["text"],
                "image_path": f"images/vn_admin_{doc_type}_{i+1:03d}.png",
                "metadata": {
                    "language": "vietnamese",
                    "document_type": doc_type,
                    "document_name": template_info["name"],
                    "created_date": self._random_date().isoformat(),
                    **doc_content["metadata"]
                }
            }
            
            documents.append(doc)
        
        self.logger.info(f"Generated {len(documents)} document records")
        return documents
    
    def _generate_document_content(self, doc_type: str, template_info: Dict) -> Dict[str, Any]:
        """Generate content for a specific document type."""
        template = template_info["template"]
        metadata_template = template_info["metadata_template"]
        
        # Generate random values for template variables
        values = {}
        
        if doc_type == "cong_van":
            values = {
                "so_van_ban": f"{random.randint(100, 999)}",
                "phong_ban": random.choice(metadata_template["phong_ban"]),
                "tieu_de": random.choice(metadata_template["subjects"]),
                "noi_nhan": "Các trường THPT trên địa bàn thành phố",
                "noi_dung": self._generate_content_paragraph(),
                "nguoi_ky": random.choice(metadata_template["nguoi_ky"])
            }
            
        elif doc_type == "thong_bao":
            phuong = random.choice(metadata_template["phuong"])
            quan = random.choice(metadata_template["quan"])
            values = {
                "phuong": phuong,
                "quan": quan,
                "tieu_de": random.choice(metadata_template["subjects"]),
                "so_quyet_dinh": f"{random.randint(1000, 9999)}/QĐ-UBND",
                "noi_dung": self._generate_content_paragraph(),
                "dia_chi": f"Số {random.randint(1, 200)} đường {random.choice(['Láng', 'Giải Phóng', 'Hoàng Quốc Việt'])}, phường {phuong}",
                "dien_thoai": f"024.{random.randint(1000, 9999)}.{random.randint(100, 999)}",
                "chu_tich": random.choice(metadata_template["chu_tich"])
            }
            
        elif doc_type == "bao_cao":
            values = {
                "chu_de": random.choice(metadata_template["chu_de"]),
                "thoi_gian": random.choice(metadata_template["thoi_gian"]),
                "tinh_hình_chung": self._generate_content_paragraph(),
                "ket_qua": self._generate_content_paragraph(),
                "ton_tai": self._generate_content_paragraph(),
                "dinh_huong": self._generate_content_paragraph(),
                "giam_doc": random.choice(metadata_template["giam_doc"])
            }
            
        elif doc_type == "quyet_dinh":
            values = {
                "so_quyet_dinh": f"{random.randint(100, 999)}",
                "tieu_de": random.choice(metadata_template["subjects"]),
                "can_cu_phap_ly": "Nghị định số 15/2021/NĐ-CP",
                "de_nghi": random.choice(["Sở Xây dựng", "Sở Tài chính", "Sở Kế hoạch và Đầu tư"]),
                "dieu_1": self._generate_content_paragraph(),
                "dieu_2": "Các sở, ban, ngành, UBND các quận, huyện, thị xã chịu trách nhiệm thi hành quyết định này.",
                "chu_tich": random.choice(metadata_template["chu_tich"])
            }
        
        # Format template with values
        formatted_text = template.format(**values)
        
        # Extract title
        title_parts = []
        if "tieu_de" in values:
            title_parts.append(values["tieu_de"])
        if "chu_de" in values:
            title_parts.append(values["chu_de"])
        
        title = " - ".join(title_parts) if title_parts else f"{template_info['name']} số {values.get('so_van_ban', random.randint(1, 999))}"
        
        return {
            "text": formatted_text.strip(),
            "title": title,
            "metadata": {k: v for k, v in values.items() if k in ["so_van_ban", "so_quyet_dinh", "nguoi_ky", "chu_tich", "giam_doc"]}
        }
    
    def _generate_content_paragraph(self) -> str:
        """Generate a realistic Vietnamese administrative content paragraph."""
        templates = [
            "Thực hiện chỉ đạo của {co_quan_cap_tren}, trong thời gian qua, {don_vi} đã tập trung triển khai các biện pháp {hoat_dong}. Kết quả đạt được là {ket_qua_tich_cuc}, tuy nhiên vẫn còn một số {van_de_ton_tai} cần được khắc phục.",
            
            "Căn cứ vào tình hình thực tế và yêu cầu {muc_tieu}, {don_vi} đề xuất {giai_phap} nhằm nâng cao {hieu_qua}. Việc thực hiện cần sự phối hợp chặt chẽ của {cac_ben_lien_quan}.",
            
            "Trong {thoi_gian}, {don_vi} đã {hanh_dong} và đạt được {thanh_tich}. Để tiếp tục duy trì {ket_qua_dat_duoc}, cần có {ke_hoach_tiep_theo} phù hợp với {dieu_kien_thuc_te}."
        ]
        
        # Sample values for template variables
        values = {
            "co_quan_cap_tren": random.choice(["UBND thành phố", "Bộ Giáo dục và Đào tạo", "Thành ủy Hà Nội"]),
            "don_vi": random.choice(["đơn vị", "sở", "phòng", "cơ quan"]),
            "hoat_dong": random.choice(["nâng cao chất lượng", "đẩy mạnh cải cách", "tăng cường kiểm tra"]),
            "ket_qua_tich_cuc": random.choice(["hiệu quả rõ rệt", "kết quả khả quan", "những tiến bộ đáng kể"]),
            "van_de_ton_tai": random.choice(["hạn chế", "khó khăn", "vướng mắc"]),
            "muc_tieu": random.choice(["phát triển bền vững", "nâng cao hiệu quả", "cải thiện chất lượng"]),
            "giai_phap": random.choice(["các biện pháp cụ thể", "kế hoạch hành động", "giải pháp đồng bộ"]),
            "hieu_qua": random.choice(["chất lượng công việc", "hiệu quả hoạt động", "năng lực thực hiện"]),
            "cac_ben_lien_quan": random.choice(["các đơn vị liên quan", "tất cả các bên", "các cơ quan chức năng"]),
            "thoi_gian": random.choice(["thời gian qua", "giai đoạn vừa qua", "những tháng gần đây"]),
            "hanh_dong": random.choice(["tập trung chỉ đạo", "triển khai đồng bộ", "thực hiện nghiêm túc"]),
            "thanh_tich": random.choice(["kết quả tích cực", "những thành tựu quan trọng", "hiệu quả đáng ghi nhận"]),
            "ket_qua_dat_duoc": random.choice(["thành quả", "kết quả", "hiệu quả"]),
            "ke_hoach_tiep_theo": random.choice(["định hướng phù hợp", "kế hoạch cụ thể", "các bước tiếp theo"]),
            "dieu_kien_thuc_te": random.choice(["tình hình hiện tại", "điều kiện cụ thể", "thực tiễn địa phương"])
        }
        
        template = random.choice(templates)
        return template.format(**values)
    
    def _random_date(self) -> datetime:
        """Generate a random date within the last year."""
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        return start_date + timedelta(days=random_days)
    
    def _create_document_images(self, documents: List[Dict]):
        """Create image files for documents (requires PIL)."""
        if not PIL_AVAILABLE:
            return
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for doc in documents:
            img_path = self.images_dir / Path(doc["image_path"]).name
            
            # Create a simple document image
            img_width, img_height = 800, 1000
            img = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(img)
            
            # Add document text with word wrapping
            text = doc["text"]
            lines = []
            words = text.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if len(test_line) < 80:  # Approximate character limit per line
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw text lines
            y_position = 50
            line_height = 20
            
            for line in lines[:45]:  # Limit to first 45 lines
                draw.text((50, y_position), line, fill='black', font=font)
                y_position += line_height
                
                if y_position > img_height - 50:
                    break
            
            # Save image
            img.save(img_path)
            
            # Also save OCR text file
            ocr_path = self.ocr_dir / f"{Path(img_path).stem}.txt"
            with open(ocr_path, 'w', encoding='utf-8') as f:
                f.write(doc["text"])
        
        self.logger.info(f"Created {len(documents)} document images")
    
    def _create_text_files(self, documents: List[Dict]):
        """Create text files when PIL is not available."""
        for doc in documents:
            txt_path = self.images_dir / f"{doc['doc_id']}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(doc["text"])
            
            # Update image path to point to text file
            doc["image_path"] = f"images/{doc['doc_id']}.txt"
        
        self.logger.info(f"Created {len(documents)} text files")
    
    def _create_corpus_file(self, documents: List[Dict]):
        """Create corpus.jsonl file."""
        corpus_path = self.output_path / "corpus.jsonl"
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                corpus_entry = {
                    "_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "image_path": doc["image_path"],
                    "metadata": doc["metadata"]
                }
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created corpus file: {corpus_path}")
    
    def _create_queries_file(self, documents: List[Dict]):
        """Create queries.jsonl file with realistic Vietnamese queries."""
        queries_path = self.output_path / "queries.jsonl"
        
        # Generate diverse queries based on document content
        queries = []
        query_id = 1
        
        # Document type based queries
        for doc_type in self.DOCUMENT_TEMPLATES.keys():
            doc_name = self.DOCUMENT_TEMPLATES[doc_type]["name"]
            queries.extend([
                {"query_id": f"q{query_id:03d}", "text": f"Tìm {doc_name.lower()} về giáo dục"},
                {"query_id": f"q{query_id+1:03d}", "text": f"{doc_name} mới nhất"},
                {"query_id": f"q{query_id+2:03d}", "text": f"Danh sách {doc_name.lower()}"}
            ])
            query_id += 3
        
        # Content-based queries
        content_queries = [
            "thông tin về tuyển sinh",
            "quy định mới về giáo dục",
            "hướng dẫn thủ tục hành chính",
            "lịch họp và hội nghị",
            "báo cáo tình hình thực hiện",
            "quyết định phê duyệt dự án",
            "thông báo về cắt điện nước",
            "công văn chỉ đạo mới",
            "kế hoạch năm học mới",
            "quy chế tổ chức hoạt động"
        ]
        
        for content in content_queries:
            queries.append({
                "query_id": f"q{query_id:03d}",
                "text": content
            })
            query_id += 1
        
        # Specific administrative queries
        admin_queries = [
            "văn bản từ Sở Giáo dục",
            "thông báo từ UBND phường",
            "báo cáo từ Sở Tài nguyên",
            "quyết định từ UBND thành phố",
            "công văn có số văn bản 100",
            "tài liệu về môi trường",
            "văn bản ký bởi Giám đốc",
            "thông báo khẩn cấp",
            "báo cáo quý 1",
            "quyết định mới ban hành"
        ]
        
        for admin_query in admin_queries:
            queries.append({
                "query_id": f"q{query_id:03d}",
                "text": admin_query
            })
            query_id += 1
        
        # Write queries file
        with open(queries_path, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created {len(queries)} queries in: {queries_path}")
        return queries
    
    def _create_qrels_file(self, documents: List[Dict]):
        """Create qrels.jsonl file with relevance judgments."""
        qrels_path = self.output_path / "qrels.jsonl"
        
        # Load queries
        queries_path = self.output_path / "queries.jsonl"
        queries = []
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line))
        
        qrels_data = {}
        
        # Create relevance judgments based on keyword matching
        for query in queries:
            query_id = query["query_id"]
            query_text = query["text"].lower()
            
            qrels_data[query_id] = {}
            
            for doc in documents:
                doc_id = doc["doc_id"]
                relevance = 0
                
                # Check for direct keyword matches
                doc_text_lower = doc["text"].lower()
                title_lower = doc["title"].lower()
                doc_type = doc["document_type"]
                
                # High relevance (2) - exact matches
                if any(keyword in query_text for keyword in [doc_type, doc["metadata"]["document_name"].lower()]):
                    if any(keyword in doc_text_lower or keyword in title_lower 
                          for keyword in query_text.split() if len(keyword) > 3):
                        relevance = 2
                
                # Medium relevance (1) - partial matches
                elif any(keyword in doc_text_lower or keyword in title_lower 
                        for keyword in query_text.split() if len(keyword) > 3):
                    relevance = 1
                
                # Document type specific matching
                if any(doc_type_word in query_text 
                      for doc_type_word in ["công văn", "thông báo", "báo cáo", "quyết định"]):
                    if doc_type in query_text or doc["metadata"]["document_name"].lower() in query_text:
                        relevance = max(relevance, 2)
                
                # Organization matching
                if "sở giáo dục" in query_text and "giáo dục" in doc_text_lower:
                    relevance = max(relevance, 1)
                if "ubnd" in query_text and "ubnd" in doc_text_lower:
                    relevance = max(relevance, 1)
                
                # Only include relevant documents
                if relevance > 0:
                    qrels_data[query_id][doc_id] = relevance
        
        # Write qrels file
        with open(qrels_path, 'w', encoding='utf-8') as f:
            for query_id, doc_rels in qrels_data.items():
                for doc_id, relevance in doc_rels.items():
                    qrel_entry = {
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "relevance": relevance
                    }
                    f.write(json.dumps(qrel_entry, ensure_ascii=False) + '\n')
        
        total_qrels = sum(len(doc_rels) for doc_rels in qrels_data.values())
        self.logger.info(f"Created {total_qrels} relevance judgments in: {qrels_path}")
    
    def _create_metadata_file(self, documents: List[Dict]):
        """Create metadata.json file."""
        metadata_path = self.output_path / "metadata.json"
        
        # Count documents by type
        doc_type_counts = {}
        for doc in documents:
            doc_type = doc["document_type"]
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        
        metadata = {
            "dataset_name": "Vietnamese Administrative Documents Sample",
            "description": "Sample dataset of Vietnamese administrative documents for testing and demonstration",
            "language": "vietnamese",
            "created_date": datetime.now().isoformat(),
            "version": "1.0.0",
            "total_documents": len(documents),
            "document_types": doc_type_counts,
            "document_type_descriptions": {
                doc_type: info["name"] 
                for doc_type, info in self.DOCUMENT_TEMPLATES.items()
            },
            "format": {
                "corpus": "jsonl",
                "queries": "jsonl", 
                "qrels": "jsonl",
                "images": "png" if PIL_AVAILABLE else "txt"
            },
            "fields": {
                "corpus": ["_id", "title", "text", "image_path", "metadata"],
                "queries": ["query_id", "text"],
                "qrels": ["query_id", "doc_id", "relevance"]
            },
            "statistics": {
                "avg_document_length": sum(len(doc["text"].split()) for doc in documents) / len(documents),
                "avg_title_length": sum(len(doc["title"].split()) for doc in documents) / len(documents),
                "character_encoding": "utf-8"
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Created metadata file: {metadata_path}")


def main():
    """CLI entry point for Vietnamese sample creator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Vietnamese Administrative Documents Sample Dataset")
    parser.add_argument("output_path", help="Output directory for sample dataset")
    parser.add_argument("--num-docs", type=int, default=50, help="Number of documents to create")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create sample dataset
    creator = VietnameseSampleCreator(args.output_path)
    success = creator.create_sample_dataset(args.num_docs)
    
    if success:
        print(f"\n✓ Vietnamese sample dataset created successfully!")
        print(f"Location: {args.output_path}")
        print(f"Documents: {args.num_docs}")
        print("\nFiles created:")
        print("  - corpus.jsonl    (document corpus)")
        print("  - queries.jsonl   (search queries)")
        print("  - qrels.jsonl     (relevance judgments)")
        print("  - metadata.json   (dataset metadata)")
        print("  - images/         (document images or text files)")
        if PIL_AVAILABLE:
            print("  - ocr/            (OCR text files)")
    else:
        print("✗ Failed to create sample dataset")
        exit(1)


if __name__ == "__main__":
    main()
