#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天龙八部小说记忆构建系统
基于MemOS框架构建小说人物记忆立方体
"""

import requests
import json
import os
import pickle
import time
import uuid
import logging
from datetime import datetime

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JSON修复功能配置
try:
    from json_repair import repair_json
    HAS_JSONREPAIR = True
    logger.info("✓ jsonrepair库已加载，JSON修复功能已启用")
except ImportError:
    HAS_JSONREPAIR = False
    logger.warning("⚠ jsonrepair库未安装，将使用基础修复策略")
    def repair_json(text):
        return text

class TaskType(Enum):
    """任务类型枚举"""
    EVENT_EXTRACTION = "event_extraction"
    CHARACTER_ANALYSIS = "character_analysis"
    PLOT_SPECULATION = "plot_speculation"

class MemOSLLMClient:
    """对话客户端 - 使用MemOS让AI调用变得简单可靠"""
  
    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        """初始化MemOS LLM客户端"""
        try:
            # 导入MemOS的智能组件
            from memos.llms.factory import LLMFactory
            from memos.configs.llm import LLMConfigFactory
        except ImportError as e:
            logger.error(f"MemOS导入失败: {e}")
            raise ImportError("请确保已正确安装MemOS库")

        # 配置LLM
        llm_config_factory = LLMConfigFactory(
            backend="qwen",
            config={
                "model_name_or_path": model,
                "api_key": api_key,
                "api_base": api_base,
                "temperature": 0.8,
                "max_tokens": 8192,
                "top_p": 0.9,
            }
        )

        # 创建对话客户端
        self.llm = LLMFactory.from_config(llm_config_factory)
        logger.info(f"对话客户端已就绪！使用模型: {model}")

    def call_api(self, messages: List[Dict], task_type: TaskType, timeout: int = 1800) -> Dict:
        """和AI对话的方法"""
        try:
            response = self.llm.generate(messages)
            return {
                "status": "success",
                "content": response,
                "model_used": self.llm.config.model_name_or_path
            }
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_used": self.llm.config.model_name_or_path
            }


class Prompt:
    """提示词管理类 - 包含所有静态方法"""

    @staticmethod
    def extract_character_names_prompt(paragraph: str, alias_to_name: dict = None):
        """人物识别提示词"""
        system_msg = (
            "你是一个小说人物识别专家，请从以下小说片段中提取所有明确提及的人物。\n"
            "对于每个人物，请标注该人物的标准姓名（如\"乔峰\"）以及其在该片段中出现的所有称呼、别名、代称（如\"丐帮帮主\"、\"乔帮主\"、\"那大汉\"）。\n\n"
            "请以如下格式返回 JSON：\n"
            "[\n"
            "  {\n"
            "    \"name\": \"乔峰\",\n"
            "    \"aliases\": [\"丐帮帮主\", \"乔帮主\", \"那大汉\"]\n"
            "  }\n"
            "]\n\n"
            "注意：\n"
            "1. 只包含人物，不包括地点或组织。\n"
            "2. 同一人物的多个称呼应统一归并在同一个条目中。\n"
            "3. 所有字段使用标准 JSON 格式。不要包含 markdown 符号或注释。\n"
            "4. 如果无法确定某个称呼是否为新人物，可以暂时保留为独立项。"
        )

        if alias_to_name:
            system_msg += "\n\n以下是已知别名对应的标准人物姓名，请尽量将新识别到的称呼归入已有人物中：\n"
            alias_map_str = json.dumps(alias_to_name, ensure_ascii=False, indent=2)
            system_msg += alias_map_str

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"小说片段如下：\n{paragraph}"}
        ]

    @staticmethod
    def update_characters_batch_prompt(characters_data: dict, paragraph: str):
        """批量人物更新提示词"""
        # 构建人物信息字符串
        characters_info = []
        for name, memcube in characters_data.items():
            unfinished_events = [event for event in memcube.get("events", []) if not event.get("if_completed", False)]
            characters_info.append(f"人物姓名：{name}")
            characters_info.append(f"当前未完结事件：{json.dumps(unfinished_events, ensure_ascii=False, indent=2)}")
            characters_info.append("---")
        
        return [
            {
                "role": "system",
                "content": (
                    "你是小说人物建模专家，将分析多个人物的未完成事件与最新小说片段。\n"
                    "你的任务是为每个人物更新以下字段：\n"
                    "- events：事件列表，每个事件包含以下子字段：\n"
                    "  * event_id：唯一标识符（如 \"event_001\"）\n"
                    "  * action：人物在此事件中的具体行为/动作（如 \"寻找\"、\"决斗\"、\"逃跑\"）\n"
                    "  * event：事件的整体描述，可能涉及多个人物（如 \"张三与李四在酒楼发生冲突\"）\n"
                    "  * motivation：该人物行动的动机/原因\n"
                    "  * impact：事件对该人物的影响\n"
                    "  * involved_entities：参与此事件的所有人物列表\n"
                    "  * time：事件发生的时间\n"
                    "  * location：事件发生的地点\n"
                    "  * if_completed：事件是否已完结（true/false）\n\n"
                    "- utterances：该人物说过的话（含时间或事件编号）\n"
                    "- speech_style：说话风格（如 古典、直接、讽刺等）\n"
                    "- personality_traits：性格特征列表（如 [\"冷静\", \"谨慎\"]）\n"
                    "- emotion_state：当前情绪状态\n"
                    "- relations：与他人的关系列表\n\n"
                    "字段说明示例：\n"
                    "如果事件是\"乔峰在聚贤庄与众武林人士激战\"，那么：\n"
                    "- action: \"激战\"（乔峰的具体行为）\n"
                    "- event: \"乔峰在聚贤庄与众武林人士激战\"（完整事件描述）\n\n"
                    "请特别注意以下要求：\n"
                    "1. 请认真判断现有未完成事件是否已经在新片段中结束。\n"
                    "2. 如果某事件已有结局或结果，请务必将其 `if_completed` 字段标记为 true。\n"
                    "3. 如果小说片段中出现与该人物相关的新事件，请添加新的事件条目。\n"
                    "4. 只更新在本章节中有出现或提及的人物，没有出现的人物可以返回空的更新。\n"
                    "最终请输出以下 JSON 结构（包含所有人物的更新信息）：\n"
                    "{\n"
                    "  \"人物名1\": {\n"
                    "    \"events\": [...],\n"
                    "    \"utterances\": [...],\n"
                    "    \"speech_style\": \"...\",\n"
                    "    \"personality_traits\": [...],\n"
                    "    \"emotion_state\": \"...\",\n"
                    "    \"relations\": [...]\n"
                    "  },\n"
                    "  \"人物名2\": {\n"
                    "    \"events\": [...],\n"
                    "    \"utterances\": [...],\n"
                    "    \"speech_style\": \"...\",\n"
                    "    \"personality_traits\": [...],\n"
                    "    \"emotion_state\": \"...\",\n"
                    "    \"relations\": [...]\n"
                    "  }\n"
                    "}\n\n"
                    "请注意：\n"
                    "1. 所有字段名必须使用双引号包裹（JSON 标准格式）。\n"
                    "2. 不要添加注释符号、额外说明或 markdown 符号。\n"
                    "3. 仅返回完整 JSON 对象，不能是数组或其他格式。\n"
                    "4. 如某个人物没有内容可填，请使用空数组 [] 或空字符串 \"\"。\n"
                    "5. 如某个人物在本章节中完全没有出现，可以省略该人物或返回空的更新信息。\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"需要更新的人物信息：\n\n"
                    f"{chr(10).join(characters_info)}\n\n"
                    f"小说片段如下：\n{paragraph}\n\n"
                    "请按上述格式返回所有人物的更新信息。"
                )
            }
        ]

    @staticmethod
    def speculate_event_outcome(character_name: str, memcube: dict, user_input: str):
        """基于人物记忆的情节推演 - 生成小说风格的叙述"""
        return [
            {
                "role": "system",
                "content": (
                    "你是一位小说剧本推演专家。\n"
                    "你将获得所有人物的完整 JSON 信息（包括事件链、性格、情绪、关系等）以及一条用户提出的假设性情节。\n"
                    "你的任务是基于这些人物的背景、未完成事件、关系网、性格与动机，合理地推演故事可能的发展。\n"
                    "请生成完整的小说段落风格的叙述（不是列表、不是 JSON），描写故事如何展开。\n"
                    "注意语言风格应与原小说保持一致（如古典武侠风）。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"人物姓名：{character_name}\n\n"
                    f"人物信息如下（JSON 格式）：\n{json.dumps(memcube, ensure_ascii=False, indent=2)}\n\n"
                    f"用户的假设性情节如下：\n{user_input}\n\n"
                    "请基于上述信息推演故事的发展，返回小说式语言的叙述，不要包含任何解释性语言或 JSON。"
                )
            }
        ]

    @staticmethod
    def evaluate_plot_reasonableness(character_name: str, memcube: dict, user_input: str):
        """剧情合理性分析 - 基于人物逻辑判断情节可信度"""
        return [
            {
                "role": "system",
                "content": (
                    "你是一个小说人物行为合理性分析专家。\n"
                    "你将获得所有人物的完整 JSON 信息（包括事件链、性格、情绪、关系等）以及用户提出的一条假设性剧情。\n"
                    "你的任务是：\n"
                    "1. 判断该剧情是否符合该人物的行为逻辑、性格特征、情绪状态以及当前背景。\n"
                    "2. 如不合理，请指出具体不合理的地方，并说明原因。\n"
                    "3. 如合理，请说明其合理性，并简要描述该剧情如何顺理成章地发生。\n\n"
                    "返回格式：\n"
                    "- 合理性评估：合理 / 不合理 / 有条件合理\n"
                    "- 分析说明：详细解释是否符合人物动机、关系与背景\n"
                    "- 建议：如有必要，提出修改建议或更合理的替代表达\n\n"
                    "请用简洁中文回答，不要生成小说正文或 JSON 结构。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"人物姓名：{character_name}\n\n"
                    f"所有人物的完整信息如下（JSON 格式）：\n{json.dumps(memcube, ensure_ascii=False, indent=2)}\n\n"
                    f"用户提出的剧情设想如下：\n{user_input}\n\n"
                    "请你判断这个剧情是否符合该人物当前的状态与逻辑，并说明理由。"
                )
            }
        ]

    @staticmethod
    def emotion_trajectory_prompt(character_name: str, memcube: dict, user_input: str):
        """情绪轨迹分析 - 预测人物情绪变化"""
        return [
            {
                "role": "system",
                "content": (
                    "你是一个小说人物情绪轨迹分析专家。\n"
                    "你将获得某个人物的完整信息（包括事件、性格、情绪、关系等）和用户设想的一段剧情。\n"
                    "请判断在该剧情中，该人物的情绪是否会发生变化。\n\n"
                    "你的任务是：\n"
                    "1. 判断该剧情设想中是否包含情绪变化。\n"
                    "2. 如果有，请指出情绪类型，并解释该变化是如何被激发的。\n"
                    "3. 如果没有，请说明为何情绪保持稳定。\n\n"
                    "返回格式：\n"
                    "- 情绪变化：有 / 无\n"
                    "- 当前情绪：xxx\n"
                    "- 变化原因：xxx\n"
                    "请使用简洁中文作答。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"人物姓名：{character_name}\n\n"
                    f"该人物的完整信息如下（JSON）：\n{json.dumps(memcube, ensure_ascii=False, indent=2)}\n\n"
                    f"用户设想剧情如下：\n{user_input}"
                )
            }
        ]

    @staticmethod
    def conflict_progression_prompt(character_name: str, memcube: dict, user_input: str):
        """冲突演化分析 - 追踪人物间矛盾发展"""
        return [
            {
                "role": "system",
                "content": (
                    "你是一个小说人物之间矛盾关系演化的分析专家。\n"
                    "你将获得某人物的完整资料（JSON 格式）以及用户提出的一条设想剧情。\n"
                    "请判断在这条剧情中，是否涉及与他人之间的矛盾进展。\n\n"
                    "你的任务是：\n"
                    "1. 判断该设想剧情中是否涉及已有或潜在冲突对象。\n"
                    "2. 如果有，请判断该关系是否发生变化（如激化、缓和或解决）。\n"
                    "3. 简述冲突变化的原因。\n\n"
                    "返回格式：\n"
                    "- 对手：xxx\n"
                    "- 当前阶段：xxx（如：潜在 → 升级 → 缓和 → 解决）\n"
                    "- 变化原因：xxx\n"
                    "请使用简洁中文作答。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"人物姓名：{character_name}\n\n"
                    f"该人物的完整信息如下（JSON）：\n{json.dumps(memcube, ensure_ascii=False, indent=2)}\n\n"
                    f"用户设想剧情如下：\n{user_input}"
                )
            }
        ]


class NovelMemoryBuilder:
    """小说记忆构建器"""

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        """初始化记忆构建器"""
        self.api_client = MemOSLLMClient(api_key, api_base, model)
        self.memcubes = {}  # 全局人物记忆库
        self.alias_to_name = {}  # 别名到标准名映射

    def extract_all_chapters(self, text: str, output_dir: str = "chapters"):
        """提取所有章节"""
        # 匹配所有"第X章"标题的位置
        pattern = r"(第[一二三四五六七八九十百千零〇两\d]+回)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            raise ValueError("未找到任何章节标题")

        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(matches)):
            start_idx = matches[i].start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapter_title = matches[i].group()
            chapter_number = i + 1  # 用自然数编号

            chapter_text = text[start_idx:end_idx].strip()
            filename = os.path.join(output_dir, f"chapter{chapter_number}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chapter_text)
            logger.info(f"已保存：{filename}（{chapter_title}）")

    def init_memcube(self, character_name: str, chunk_id: str):
        """初始化人物记忆立方体 - 包含所有核心字段"""
        return {
            "name": character_name,
            "first_appearance": chunk_id,
            "aliases": [character_name],
            "events": [],
            "utterances": [],
            "speech_style": "",
            "personality_traits": [],
            "emotion_state": "",
            "relations": []
        }

    def get_unfinished_events(self, memcube: dict):
        """获取未完成的事件列表 - 用于上下文连续性"""
        return [event for event in memcube.get("events", []) if not event.get("if_completed", False)]

    def merge_events(self, old_events: list, new_events: list):
        """智能事件合并 - 处理状态更新和新增事件"""
        event_dict = {e["event_id"]: e for e in old_events}

        for new_event in new_events:
            eid = new_event["event_id"]
            if eid in event_dict:
                # 合并策略：新字段优先，保留历史信息
                merged = event_dict[eid].copy()
                for key, value in new_event.items():
                    if value not in [None, "", []]:
                        merged[key] = value
                event_dict[eid] = merged
            else:
                event_dict[eid] = new_event  # 新事件直接加入

        return list(event_dict.values())

    def merge_unique_list(self, old: list, new: list):
        """列表去重合并 - 保持原有顺序"""
        combined = old + new
        seen = set()
        result = []
        for item in combined:
            if isinstance(item, dict):
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            else:
                key = str(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def update_memcubes_batch(self, characters_data: dict, content: str, chapter_id: str):
        """批量更新多个人物的记忆立方体"""
        try:
            # 构建批量更新提示
            update_prompt = Prompt.update_characters_batch_prompt(characters_data, content)
            
            # 调用API
            result = self.api_client.call_api(update_prompt, TaskType.CHARACTER_ANALYSIS, timeout=1800)
            
            if result["status"] != "success":
                return None, result.get("error", "API调用失败")
            
            # 解析结果
            try:
                content_str = result.get("content", "").strip()
                # 清理可能的markdown标记
                content_str = content_str.strip("```json").strip("```").strip()
                updated_data = json.loads(content_str)
                return updated_data, None
            except json.JSONDecodeError as e:
                # 尝试修复JSON
                if HAS_JSONREPAIR:
                    try:
                        repaired = repair_json(content_str)
                        updated_data = json.loads(repaired)
                        return updated_data, None
                    except:
                        pass
                return None, f"JSON解析失败: {e}"
                
        except Exception as e:
            logger.error(f"批量更新人物时发生错误: {e}")
            return None, str(e)

    def process_chapters(self, chapter_folder: str = "chapters"):
        """处理所有章节"""
        # 按章节顺序处理
        chapter_files = sorted(
            [os.path.join(chapter_folder, f) for f in os.listdir(chapter_folder) 
             if f.startswith("chapter") and f.endswith(".txt")],
            key=lambda x: int(re.search(r'chapter(\d+)', x).group(1))
        )

        for chapter_file in chapter_files:
            chapter_id = os.path.basename(chapter_file).replace(".txt", "")
            logger.info(f"\n 正在处理：{chapter_id}")
      
            with open(chapter_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 执行人物识别和初始化
            self._process_character_identification(content, chapter_id)
            
            # 更新所有人物状态
            self._update_all_characters(content, chapter_id)

    def _process_character_identification(self, content: str, chapter_id: str):
        """处理人物识别"""
        name_prompt = Prompt.extract_character_names_prompt(content, self.alias_to_name)
        name_result = self.api_client.call_api(name_prompt, TaskType.EVENT_EXTRACTION, timeout=1800)

        try:
            content_str = name_result.get("content", "").strip("```json").strip("```").strip()
            extracted = json.loads(content_str)
        except:
            extracted = []

        # 更新人物库和别名映射
        for item in extracted:
            std_name = item["name"]
            aliases = item.get("aliases", [])
        
            # 初始化或更新 MemCube
            if std_name not in self.memcubes:
                logger.info(f"新人物识别：{std_name}")
                self.memcubes[std_name] = self.init_memcube(std_name, chapter_id)
                self.memcubes[std_name]["aliases"] = []

            # 合并别名列表
            all_aliases = list(set(self.memcubes[std_name].get("aliases", []) + aliases))
            self.memcubes[std_name]["aliases"] = all_aliases

            # 构建全局别名映射
            for alias in [std_name] + aliases:
                self.alias_to_name[alias] = std_name

    def _update_all_characters(self, content: str, chapter_id: str):
        """批量更新所有人物状态"""
        if not self.memcubes:
            logger.info("没有人物需要更新")
            return
            
        try:
            # 批量调用API更新所有人物
            updated_data, error = self.update_memcubes_batch(self.memcubes, content, chapter_id)
            
            if error:
                logger.warning(f"⚠️ 批量更新失败 in {chapter_id} -> {error}")
                return
                
            if not updated_data:
                logger.warning(f"⚠️ 批量更新返回空结果 in {chapter_id}")
                return

            # 处理批量更新结果
            for name, updated in updated_data.items():
                if name not in self.memcubes:
                    logger.warning(f"⚠️ 返回了未知人物：{name}")
                    continue
                    
                if not updated:
                    logger.info(f"📝 {name} 在本章节中无更新")
                    continue

                # 智能合并更新结果
                memcube = self.memcubes[name]
                memcube["events"] = self.merge_events(memcube["events"], updated.get("events", []))
                memcube["utterances"].extend(updated.get("utterances", []))
                if updated.get("speech_style"):
                    memcube["speech_style"] = updated["speech_style"]
                memcube["personality_traits"] = self.merge_unique_list(
                    memcube["personality_traits"], updated.get("personality_traits", [])
                )
                if updated.get("emotion_state"):
                    memcube["emotion_state"] = updated["emotion_state"]
                memcube["relations"].extend(updated.get("relations", []))
                
                logger.info(f"✅ 成功更新：{name}")

        except Exception as e:
            logger.error(f"⚠️ 批量更新异常 in {chapter_id} -> {e}")

    def save_memcubes(self, filename: str = "memcubes.json"):
        """保存记忆立方体"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.memcubes, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 记忆立方体已保存到 {filename}")

    def load_memcubes(self, filename: str = "memcubes.json"):
        """加载记忆立方体"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.memcubes = json.load(f)
            logger.info(f"✅ 记忆立方体已从 {filename} 加载")
        except FileNotFoundError:
            logger.warning(f"⚠️ 文件 {filename} 不存在")

class MemoryConverter:
    """记忆格式转换器"""
    
    @staticmethod
    def create_memory_node(content: str, entities: list, key: str, memory_type: str = "fact") -> dict:
        """创建标准化的Memory节点"""
        node_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
      
        # 模拟embedding（实际应用中应使用真实的embedding服务）
        embedding = [0.1] * 768  # 示例维度
      
        return {
            "id": node_id,
            "memory": content,
            "metadata": {
                "user_id": "",
                "session_id": "",
                "status": "activated",
                "type": "fact",
                "confidence": 0.99,
                "entities": entities,
                "tags": ["事件"] if "事件" in key else ["关系"],
                "updated_at": now,
                "memory_type": memory_type,
                "key": key,
                "sources": [],
                "embedding": embedding,
                "created_at": now,
                "usage": [],
                "background": ""
            }
        }

    @staticmethod
    def convert_memcube_to_memory_graph(input_file: str, output_file: str):
        """将MemCube格式转换为Memory Graph格式"""
        with open(input_file, "r", encoding="utf-8") as f:
            memcube_data = json.load(f)

        nodes = []
        edges = []

        for character, data in memcube_data.items():
            previous_event_id = None

            # === 事件序列转换 ===
            for event in data.get("events", []):
                memory_text = f"{character}在{event.get('time')}于{event.get('location')}，因为{event.get('motivation')}，进行了{event.get('action')}，结果是{event.get('impact')}。"
                entities = [character] + event.get("involved_entities", [])
                node = MemoryConverter.create_memory_node(
                    content=memory_text,
                    entities=entities,
                    key=f"{character}的事件：{event.get('action')}"
                )
                nodes.append(node)

                # 建立事件时序关系
                if previous_event_id:
                    edges.append({
                        "source": previous_event_id,
                        "target": node["id"],
                        "type": "FOLLOWS"
                    })
                previous_event_id = node["id"]

            # === 关系网络聚合 ===
            relations_texts = []
            seen = set()
            for relation in data.get("relations", []):
                name = relation.get("name") or relation.get("人物") or relation.get("character")
                relation_text = relation.get("relation") or relation.get("relationship") or relation.get("关系")
                if not name or not relation_text:
                    continue
                dedup_key = (str(name), str(relation_text))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                relations_texts.append(f"与{name}是{relation_text}")

            if relations_texts:
                memory_text = f"{character}" + "，".join(relations_texts) + "。"
                entities = [character]
                node = MemoryConverter.create_memory_node(
                    content=memory_text,
                    entities=entities,
                    key=f"{character}的关系汇总",
                )
                nodes.append(node)

        # 保存转换结果
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": nodes,
                "edges": edges
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 完成转换，共生成 {len(nodes)} 个 memory 节点，{len(edges)} 条边")
        logger.info(f"📁 输出文件: {output_file}")

class WuxiaTextGame:
    """武侠文字游戏核心类"""
    
    def __init__(self, mos_config):
        """初始化游戏"""
        try:
            from memos.mem_os.main import MOS
            
            self.world_memory = MOS(mos_config)  # 世界记忆系统
            self.character_cubes = {}            # 每个NPC的MemCube
            self.timeline_memories = []          # 时间线记忆列表
            
            # 创建游戏主用户
            self.world_memory.create_user("game_master")
            logger.info("✅ 武侠文字游戏初始化完成")
        except ImportError as e:
            logger.error(f"MemOS导入失败: {e}")
            raise
      
    def start_adventure(self, player_choice):
        """开始冒险"""
        return f"欢迎来到{player_choice.get('location', '江湖')}..."
  
    def process_action(self, player_input):
        """处理玩家的自然语言输入"""
        try:
            # 1. 理解玩家意图
            intent_analysis = self.world_memory.chat(
                query=f"分析玩家意图: {player_input}",
                user_id="game_master"
            )

            # 2. 检索相关记忆
            context = self.world_memory.search(
                query=player_input,
                user_id="game_master"
            )

            # 3. 计算行动后果
            consequences = self.predict_consequences(player_input, context)

            # 4. 更新世界状态
            self.update_world_state(player_input, consequences)

            # 5. 生成剧情发展
            return self.generate_story(player_input, context, consequences)
        except Exception as e:
            logger.error(f"处理玩家行动时发生错误: {e}")
            return "抱歉，发生了一些意外情况..."

    def predict_consequences(self, player_input, context):
        """预测玩家行动的后果"""
        query = f"基于以下背景：{context}，预测玩家行动'{player_input}'的可能后果"
        result = self.world_memory.chat(
            query=query,
            user_id="game_master"
        )
        return result

    def update_world_state(self, player_input, consequences):
        """更新世界状态到MemOS记忆中"""
        memory_content = f"玩家行动: {player_input}, 后果: {consequences}"
        self.world_memory.add(
            memory_content=memory_content,
            user_id="game_master"
        )

    def generate_story(self, player_input, context, consequences):
        """生成故事发展"""
        query = f"基于背景{context}和后果{consequences}，为玩家行动'{player_input}'生成有趣的故事发展"
        return self.world_memory.chat(
            query=query,
            user_id="game_master"
        )


def main():
    """主函数 - 演示完整的使用流程"""
    # 配置参数
    API_KEY = "sk-f9dee2f8b19f453b964cc4eb31b551bf"  # 请替换为你的API密钥
    API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen3-max"

    # 创建记忆构建器
    builder = NovelMemoryBuilder(API_KEY, API_BASE, MODEL_NAME)

    # 如果有小说文件，提取章节
    novel_file = "天龙八部.txt"
    # if os.path.exists(novel_file):
    #     logger.info("开始提取小说章节...")
    #     with open(novel_file, "r", encoding="utf-8") as f:
    #         full_text = f.read()
    #     builder.extract_all_chapters(full_text)

    # 处理章节，构建记忆
    if os.path.exists("chapters"):
        logger.info("开始构建人物记忆...")
        builder.process_chapters()

        # 保存结果
        builder.save_memcubes("memcubes_complete.json")

    # 演示情节推演
    if builder.memcubes and "段誉" in builder.memcubes:
        logger.info("🎭 演示情节推演...")
        character_name = "段誉"
        user_input = "如果段誉没有出现在剑湖宫比武大会，事情会怎样？"

        prompt = Prompt.speculate_event_outcome(character_name, builder.memcubes[character_name], user_input)
        response = builder.api_client.call_api(prompt, TaskType.PLOT_SPECULATION)
        print(f"\n🎬 情节推演结果：\n{response.get('content', ' 没有返回')}")


if __name__ == "__main__":
    main()




















