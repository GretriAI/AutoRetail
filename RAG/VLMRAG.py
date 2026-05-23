import torch
import numpy as np
from PIL import Image
import faiss  # 高性能向量检索库
from transformers import AutoProcessor, AutoModelForVision2Seq, CLIPProcessor, CLIPModel

class MultimodalRetailRAG:
    def __init__(self, vlm_model_path="Qwen/Qwen2-VL-7B-Instruct"):
        # 1. 初始化用于RAG检索的轻量级多模态向量模型 (以CLIP为例)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 2. 初始化用于终审决策的VLM大模型
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(vlm_model_path, torch_dtype=torch.float16).cuda()
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
        
        # 3. 初始化商超知识库 (FAISS 索引)
        self.dimension = 512  # CLIP-ViT-B/32 的特征维度
        self.index = faiss.IndexFlatIP(self.dimension) # 使用内积(等价于余弦相似度)
        self.sku_metadata = {}  # 存储索引ID到具体商品信息的映射 {idx: {"name": str, "price": float}}

    def add_sku_to_gallery(self, sku_id, image_path, sku_name, price):
        """往RAG知识库中注册商品（商家后台录入）"""
        img = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            feat = self.clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True) # 归一化
        
        idx = self.index.ntotal
        self.index.add(feat.cpu().numpy().astype('float32'))
        self.sku_metadata[idx] = {"sku_id": sku_id, "name": sku_name, "price": price}

    def audit_cart_item(self, yolo_crop_cv2_img, user_context=""):
        """
        核心 RAG 流水线
        yolo_crop_cv2_img: YOLO 检测并裁剪出来的商品图像
        """
        # 将 OpenCV 格式转为 PIL
        crop_pil = Image.fromarray(cv2.cvtColor(yolo_crop_cv2_img, cv2.COLOR_BGR2RGB))
        
        # ---- Step 1: 提取特征并在知识库中检索 (RAG Retrieval) ----
        inputs = self.clip_processor(images=crop_pil, return_tensors="pt").to("cuda")
        with torch.no_grad():
            query_feat = self.clip_model.get_image_features(**inputs)
            query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
            
        D, I = self.index.search(query_feat.cpu().numpy().astype('float32'), k=3) # 召回前3个最像的SKU
        
        retrieved_knowledge = ""
        for rank, idx in enumerate(I[0]):
            metadata = self.sku_metadata[idx]
            retrieved_knowledge += f"候选{rank+1}: {metadata['name']}, 售价: {metadata['price']}元。\n"

        # ---- Step 2: 构建多模态 Prompt (Generation) ----
        prompt = f"""
        你现在是XX会员店的无人结算终审系统。
        [任务描述]: 请分析监控相机裁剪出的商品图片，结合系统召回的候选商品知识库，确定顾客购物车里这件商品的准确名称。
        
        [RAG 检索出的候选商品知识库]:
        {retrieved_knowledge}
        
        [视觉推理提示]: 注意观察商品的包装细节（如克数、口味、颜色、英文字母）。
        请直接输出最终确定的商品名称和SKU ID，格式为: '商品名: <name> | ID: <id>'
        """

        # ---- Step 3: VLM 混合输入进行高精度推理 ----
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_pil},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.vlm_processor.image_processor(images=crop_pil, return_tensors="pt")
        
        vlm_inputs = self.vlm_processor(text=[text], images=crop_pil, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = self.vlm_model.generate(**vlm_inputs, max_new_tokens=100)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(vlm_inputs.input_ids, generated_ids)]
            output_text = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
        return output_text[0]
