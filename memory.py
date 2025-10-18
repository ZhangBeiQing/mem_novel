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

# 配置日志 - 同时输出到控制台和文件
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 清除默认处理器
logger.handlers.clear()

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# 文件处理器
file_handler = logging.FileHandler('memory_log.txt', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# 防止日志重复输出
logger.propagate = False

# 创建prompts目录（如果不存在）
PROMPTS_DIR = "prompts"
if not os.path.exists(PROMPTS_DIR):
    os.makedirs(PROMPTS_DIR)
    logger.info(f"✓ 创建prompts目录: {PROMPTS_DIR}")

# 更新.gitignore以忽略动态生成的prompts目录
GITIGNORE_PATH = ".gitignore"
if os.path.exists(GITIGNORE_PATH):
    with open(GITIGNORE_PATH, 'r', encoding='utf-8') as f:
        gitignore_content = f.read()
    
    if "prompts/" not in gitignore_content:
        with open(GITIGNORE_PATH, 'a', encoding='utf-8') as f:
            f.write("\n# 动态生成的AI提示词文件\nprompts/\n")
        logger.info("✓ 已将prompts/目录添加到.gitignore")

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
  
    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", model: str = "gpt-4o", backend: str = "qwen"):
        """初始化MemOS LLM客户端
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            backend: 后端类型，支持 "qwen", "openai", "claude" 等
        """
        try:
            # 导入MemOS的智能组件
            from memos.llms.factory import LLMFactory
            from memos.configs.llm import LLMConfigFactory
        except ImportError as e:
            logger.error(f"MemOS导入失败: {e}")
            raise ImportError("请确保已正确安装MemOS库")

        # 配置LLM
        llm_config_factory = LLMConfigFactory(
            backend=backend,
            config={
                "model_name_or_path": model,
                "api_key": api_key,
                "api_base": api_base,
                "temperature": 0.8,
                "max_tokens": 65536,
                "top_p": 0.9,
            }
        )

        # 创建对话客户端
        self.llm = LLMFactory.from_config(llm_config_factory)
        logger.info(f"对话客户端已就绪！使用模型: {model}")

    def call_api(self, messages: List[Dict], task_type: TaskType, timeout: int = 1800, 
                 novel_name: str = "天龙八部", chapter_id: str = "unknown", step_name: str = "default") -> Dict:
        """和AI对话的方法
        
        Args:
            messages: 对话消息列表
            task_type: 任务类型
            timeout: 超时时间
            novel_name: 小说名称，用于生成唯一prompt名称
            chapter_id: 章节ID，用于生成唯一prompt名称
            step_name: 步骤名称，用于生成唯一prompt名称
        """
        # 保存实时生成的prompt到本地JSON文件
        prompt_name = self._save_prompt_to_json(messages, task_type, novel_name, chapter_id, step_name)
        
        try:
            response = self.llm.generate(messages)
            return {
                "status": "success",
                "content": response,
                "model_used": self.llm.config.model_name_or_path,
                "prompt_name": prompt_name  # 返回生成的prompt名称
            }
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_used": self.llm.config.model_name_or_path,
                "prompt_name": prompt_name
            }
    
    def _save_prompt_to_json(self, messages: List[Dict], task_type: TaskType, 
                           novel_name: str, chapter_id: str, step_name: str) -> str:
        """保存实时生成的prompt到JSON文件，使用唯一性命名
        
        Args:
            messages: 对话消息列表
            task_type: 任务类型
            novel_name: 小说名称
            chapter_id: 章节ID
            step_name: 步骤名称
            
        Returns:
            str: 生成的prompt唯一名称
        """
        try:
            # 生成唯一的prompt名称：小说名_章节ID_任务类型_步骤名
            # 清理文件名中的特殊字符
            safe_novel_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', novel_name)
            safe_chapter_id = re.sub(r'[^\w\u4e00-\u9fff]', '_', chapter_id)
            safe_step_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', step_name)
            
            prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{task_type.value}_{safe_step_name}"
            
            # 生成文件名：唯一名称.json
            filename = f"{prompt_name}.json"
            filepath = os.path.join(PROMPTS_DIR, filename)
            
            # 构建保存的数据结构
            prompt_data = {
                "prompt_name": prompt_name,
                "novel_name": novel_name,
                "chapter_id": chapter_id,
                "step_name": step_name,
                "task_type": task_type.value,
                "timestamp": datetime.now().isoformat(),
                "model": self.llm.config.model_name_or_path,
                "messages": messages,
                "metadata": {
                    "total_messages": len(messages),
                    "system_message_length": len(messages[0]["content"]) if messages and messages[0]["role"] == "system" else 0,
                    "user_message_length": len(messages[-1]["content"]) if messages and messages[-1]["role"] == "user" else 0
                }
            }
            
            # 保存到JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Prompt已保存到: {filepath}")
            logger.info(f"✓ Prompt唯一名称: {prompt_name}")
            
            return prompt_name
            
        except Exception as e:
            logger.warning(f"保存prompt失败: {e}")  # 不影响主流程，只记录警告
            return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def call_api_with_saved_prompt(self, prompt_name: str, timeout: int = 1800) -> Dict:
        """使用已保存的prompt调用AI
        
        Args:
            prompt_name: 已保存的prompt唯一名称
            timeout: 超时时间
            
        Returns:
            Dict: AI响应结果
        """
        try:
            # 加载已保存的prompt
            filepath = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
            
            if not os.path.exists(filepath):
                logger.error(f"Prompt文件不存在: {filepath}")
                return {
                    "status": "error",
                    "error": f"Prompt文件不存在: {prompt_name}",
                    "prompt_name": prompt_name
                }
            
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            messages = prompt_data.get("messages", [])
            if not messages:
                logger.error(f"Prompt文件中没有有效的messages: {prompt_name}")
                return {
                    "status": "error",
                    "error": f"Prompt文件中没有有效的messages: {prompt_name}",
                    "prompt_name": prompt_name
                }
            
            logger.info(f"✓ 使用已保存的prompt: {prompt_name}")
            logger.info(f"✓ 小说: {prompt_data.get('novel_name', 'unknown')}")
            logger.info(f"✓ 章节: {prompt_data.get('chapter_id', 'unknown')}")
            logger.info(f"✓ 步骤: {prompt_data.get('step_name', 'unknown')}")
            logger.info(f"✓ 任务类型: {prompt_data.get('task_type', 'unknown')}")
            
            # 调用AI
            response = self.llm.generate(messages)
            return {
                "status": "success",
                "content": response,
                "model_used": self.llm.config.model_name_or_path,
                "prompt_name": prompt_name,
                "reused_prompt": True,
                "original_prompt_info": {
                    "novel_name": prompt_data.get('novel_name'),
                    "chapter_id": prompt_data.get('chapter_id'),
                    "step_name": prompt_data.get('step_name'),
                    "task_type": prompt_data.get('task_type'),
                    "original_timestamp": prompt_data.get('timestamp')
                }
            }
            
        except Exception as e:
            logger.error(f"使用已保存prompt调用API失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "prompt_name": prompt_name,
                "reused_prompt": True
            }
    
    def list_saved_prompts(self, novel_name: str = None, chapter_id: str = None, 
                          task_type: str = None) -> List[Dict]:
        """列出已保存的prompt文件
        
        Args:
            novel_name: 过滤特定小说的prompt
            chapter_id: 过滤特定章节的prompt
            task_type: 过滤特定任务类型的prompt
            
        Returns:
            List[Dict]: prompt信息列表
        """
        try:
            if not os.path.exists(PROMPTS_DIR):
                return []
            
            prompts = []
            for filename in os.listdir(PROMPTS_DIR):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(PROMPTS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        prompt_data = json.load(f)
                    
                    # 应用过滤条件
                    if novel_name and prompt_data.get('novel_name') != novel_name:
                        continue
                    if chapter_id and prompt_data.get('chapter_id') != chapter_id:
                        continue
                    if task_type and prompt_data.get('task_type') != task_type:
                        continue
                    
                    prompts.append({
                        "prompt_name": prompt_data.get('prompt_name', filename[:-5]),
                        "novel_name": prompt_data.get('novel_name', 'unknown'),
                        "chapter_id": prompt_data.get('chapter_id', 'unknown'),
                        "step_name": prompt_data.get('step_name', 'unknown'),
                        "task_type": prompt_data.get('task_type', 'unknown'),
                        "timestamp": prompt_data.get('timestamp', 'unknown'),
                        "model": prompt_data.get('model', 'unknown'),
                        "message_count": prompt_data.get('metadata', {}).get('total_messages', 0),
                        "file_path": filepath
                    })
                except Exception as e:
                    logger.warning(f"读取prompt文件失败 {filename}: {e}")
                    continue
            
            # 按时间戳排序（最新的在前）
            prompts.sort(key=lambda x: x['timestamp'], reverse=True)
            return prompts
            
        except Exception as e:
            logger.error(f"列出prompt文件失败: {e}")
            return []
    
    def search_prompts(self, keyword: str) -> List[Dict]:
        """搜索包含关键词的prompt
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            List[Dict]: 匹配的prompt信息列表
        """
        all_prompts = self.list_saved_prompts()
        keyword_lower = keyword.lower()
        
        matched_prompts = []
        for prompt in all_prompts:
            # 在各个字段中搜索关键词
            searchable_text = f"{prompt['prompt_name']} {prompt['novel_name']} {prompt['chapter_id']} {prompt['step_name']} {prompt['task_type']}".lower()
            if keyword_lower in searchable_text:
                matched_prompts.append(prompt)
        
        return matched_prompts
    
    def get_prompt_info(self, prompt_name: str) -> Dict:
        """获取特定prompt的详细信息
        
        Args:
            prompt_name: prompt唯一名称
            
        Returns:
            Dict: prompt详细信息
        """
        try:
            filepath = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
            
            if not os.path.exists(filepath):
                return {"error": f"Prompt文件不存在: {prompt_name}"}
            
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            return {
                "prompt_name": prompt_data.get('prompt_name'),
                "novel_name": prompt_data.get('novel_name'),
                "chapter_id": prompt_data.get('chapter_id'),
                "step_name": prompt_data.get('step_name'),
                "task_type": prompt_data.get('task_type'),
                "timestamp": prompt_data.get('timestamp'),
                "model": prompt_data.get('model'),
                "messages": prompt_data.get('messages', []),
                "metadata": prompt_data.get('metadata', {}),
                "file_path": filepath
            }
            
        except Exception as e:
            logger.error(f"获取prompt信息失败: {e}")
            return {"error": str(e)}


class Prompt:

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
            "1. 只包含人物或有重要剧情的动物，不包括地点或组织。\n"
            "2. 注意区分标准名称和别名、代称的区别，不要把别名代称误认为是标准名称了。同一人物的多个别称应统一归并在同一个条目中。\n"
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
                    "你的任务是为每个人物更新以下字段，对于有重要剧情的动物也要更新：\n\n"
                    "## 事件定义原则\n"
                    "事件必须是具有明确目标和结果的完整行动单元，遵循以下原则：\n"
                    "1. **目标导向性**：每个事件都有明确的目标或冲突核心\n"
                    "2. **时空边界**：在特定时间和地点发生的连续行动\n"
                    "3. **因果完整性**：包含起因、发展、高潮、结果的完整链条\n"
                    "4. **叙事重要性**：对人物发展或情节推进有实质影响\n"
                    "5. **粒度控制**：避免过细（如\"张三眨了眨眼\"）或过粗（如\"张三的一生\"）\n\n"
                    "6. **及时完结性**: 避免过粗，导致一个事件经过了几十章，夹杂了很多事件,要及时完结\n\n"
                    "## 事件边界判断标准\n"
                    "**应该合并为一个事件的情况：**\n"
                    "- 同一目标下的连续行动（如：寻找某人的整个过程）\n"
                    "- 同一冲突的完整发展（如：从争论到决斗到分胜负）\n"
                    "- 同一场景内的相关互动（如：酒楼内的完整对话和冲突）\n"
                    "**应该拆分为多个事件的情况：**\n"
                    "   - 目标发生根本改变（从寻找父亲→决定报仇）\n"
                    "   - 场景发生重大转换（从山洞→城市→寺庙）\n"
                    "   - 时间跨度过长（超过数日的连续行动）\n"
                    "   - 涉及不同的核心冲突（与A的恩怨 vs 与B的情感）\n\n"
                    "- events：字段要求\n"
                    "  * event_id：事件的唯一标识符，遵循以下规则：\n"
                    "    - 对于全新事件：使用格式 \"新事件_章节ID_序号\"（如 \"新事件_chapter1_001\"）\n"
                    "    - 每个人物的事件ID必须独立，不能与其他人物共享\n"
                    "  * action：人物在此事件中的核心行为（动词短语，如\"寻找线索\"、\"决斗较量\"、\"逃脱追捕\"）, \n"
                    "    - action是事件的核心，不要改变之前未完结事件的action，避免一个事件包含的事情越来越多。\n"
                    "    - 如果你觉得某个之前未完成的event要改action，证明它已经完成，此时你应该更新之前的未完结事件内容，把它的if_completed设为true，并重新新建一个event"
                    "  * event：事件的完整描述，必须包含：\n"
                    "    - **事件核心**：用一句话概括事件的本质和目标\n"
                    "    - **背景铺垫**：事件发生的前因后果和环境背景\n"
                    "    - **人物介绍**：涉及的关键人物及其身份、关系、特点\n"
                    "    - **发展过程**：详细描述事件的完整发展脉络，包括：\n"
                    "      * 起因：事件是如何开始的，触发因素是什么\n"
                    "      * 经过：具体的行动过程、对话内容、冲突发展\n"
                    "      * 转折：关键的转折点和意外变化\n"
                    "      * 高潮：事件的最激烈或最关键的时刻\n"
                    "    - **重要细节**：关键对话原文、具体动作描述、环境细节、情感变化\n"
                    "    - **结果状态**：事件结束时的具体状况和各方反应\n"
                    "    - **深层含义**：事件的象征意义、对后续情节的铺垫作用\n"
                    "    - 描述要求：\n"
                    "      * 使用通俗易懂的语言，避免过于简化的概括\n"
                    "      * 每个关键情节都要有具体的描述，不能只是罗列事件名称\n"
                    "      * 要让没有阅读过原著的人也能理解事件的完整过程\n"
                    "      * 重要对话要尽量保留原文或核心内容\n"
                    "  * motivation：该人物行动的深层动机（心理驱动力，而非表面原因）\n"
                    "  * impact：事件对该人物的具体影响（性格、关系、能力、地位等方面的变化）\n"
                    "  * involved_entities：参与此事件的所有重要人物列表\n"
                    "  * time：事件发生的时间范围，格式：\"第X章：具体时间背景和场景节点\"\n"
                    "  * location：事件的主要发生地点（可以是地点序列）\n"
                    "  * if_completed：事件是否已完结（true/false）\n"
                    "    - **完结标准**：事件的核心目标已达成或明确失败，且叙事焦点已转移\n"
                    "    - **未完结标准**：目标仍在追求中，或虽有阶段性结果但核心冲突未解决\n"
                    "    - 如果之前未完成的事件在当前章节完结，必须重新输出该事件并设置为true\n\n"
                    "- utterances：该人物说过的话（含时间或事件编号如：）\n"
                    "  * text：人物说的话\n"
                    "  * time：人物说话发生的绝对时间，格式要求与events中的time字段相同，必须包含章节信息和详细时间描述\n"
                    "- speech_style：说话风格（如 古典、直接、讽刺等）\n"
                    "- personality_traits：性格特征列表（如 [\"冷静\", \"谨慎\"]）\n"
                    "- emotion_state：当前情绪状态\n"
                    "- relations：人物的身份和与他人的关系列表，格式要求如下：\n"
                    "  * 必须包含人物的主要身份或职位（如门派掌门、弟子等）\n"
                    "  * 包含与其他人物的具体关系（如师父、师兄弟、对手、朋友等）\n"
                    "  * 每个关系描述应该简洁明确，体现人物在小说世界中的地位和人际网络\n"
                    "  * 示例格式：[\"无量剑西宗掌门\", \"左子穆的师妹兼对手\"]\n"
                    "  * 示例格式：[\"无量剑西宗弟子\", \"龚光杰的对手\", \"辛双清的徒弟\"]\n"
                    "  * 示例格式：[\"无量剑东宗掌门\", \"辛双清的师兄\", \"龚光杰与干光豪的师父\", \"容子矩的师兄\"]\n\n"
                    "字段说明示例：\n"
                    "按照事件定义原则，\"乔峰聚贤庄证明清白事件\"应该这样描述：\n"
                    "- event_id: \"新事件_chapter10_001\"（如果是新事件）\n"
                    "- action: \"证明清白\"（核心行为目标，而非简单的\"激战\"）\n"
                    "- event: \"乔峰聚贤庄证明清白事件：【核心目标】面对武林同道质疑，通过武力和言辞证明自己并非契丹奸细。【事件起因】身世暴露后遭到武林围攻，被迫来到聚贤庄面对众人指控；【激战过程】以一敌百展现绝世武功，击败游坦之、丁春秋等高手，用实力震慑群雄；【关键转折】在血战中展现侠义本色，救助无辜，证明品格；【结果状态】虽武力证明清白但内心痛苦，选择离开中原武林。\"\n"
                    "- motivation: \"维护名誉尊严，证明自己虽为契丹血统但仍是大宋忠义之士\"（深层心理驱动）\n"
                    "- impact: \"武功威名达到巅峰，但与中原武林彻底决裂，内心更加孤独痛苦\"（具体影响）\n"
                    "- involved_entities: [\"游坦之\", \"丁春秋\", \"聚贤庄众武林人士\"]（参与人物）\n"
                    "- time: \"第十章：大宋元佑年间，乔峰身世暴露后在聚贤庄的生死对决\"（时间范围）\n"
                    "- location: \"聚贤庄\"（主要地点）\n"
                    "- if_completed: true（目标已达成，叙事焦点已转移）\n"
                    "- relations: [\"前丐帮帮主\", \"阿朱的恋人\", \"萧远山的义子\", \"段誉和虚竹的结拜兄弟\"]（身份关系）\n\n"
                    "time字段的更多示例：\n"
                    "- 正确格式：\"第一章：大宋元佑年间，无量剑第九次比剑第四场\"\n"
                    "- 正确格式：\"第一章：大宋元佑年间，无量剑第九次比剑第四场后，钟灵放蛇时\"\n"
                    "- 正确格式：\"第二章：大宋元佑年间，段誉离开无量山后，在大理城中遇到木婉清时\"\n"
                    "- 错误格式：\"钟灵放蛇后\"（缺少章节信息和时间背景）\n"
                    "- 错误格式：\"龚光杰被貂戏弄后\"（缺少章节信息和时间背景）\n\n"
                    "请特别注意以下要求：\n"
                    "1. 请认真判断现有未完成事件是否已经在新片段中结束。\n"
                    "2. 如果某事件已有结局或结果，请务必将其 `if_completed` 字段标记为 true。\n"
                    "3. 如果小说片段中出现与该人物相关的新事件，请添加新的事件条目。\n"
                    "4. 【智能过滤】只更新在本章节中有出现或提及的人物。如某个输入人物在本章节中完全没有出现，可以省略该人物或返回空的更新信息。\n"
                    "5. 【重要】time字段必须严格按照\"第X章：详细时间描述\"的格式，绝对不能使用模糊的相对时间描述。\n"
                    "6. 【关键】事件延续性处理：\n"
                    "   - 仔细分析当前章节的事件是否是之前章节事件的延续\n"
                    "   - 如果是同一个事件的延续，必须使用相同的event_id\n"
                    "   - 在event描述中要体现完整的事件发展脉络，包括前因后果\n"
                    "   - 例如：\"段誉寻找父亲\"这个事件如果跨越多章，应该保持相同event_id，但在每章的描述中更新最新进展\n"
                    "7. 【重要】新事件识别：\n"
                    "   - 只有当事件确实是全新的、与之前事件无关的情况下，才创建新的event_id\n"
                    "   - 新事件的event_id格式：\"新事件_当前章节ID_序号\"\n"
                    "8. 【优化处理】对于在当前章节中未出现的人物：\n"
                    "   - 可以直接省略该人物，不在输出JSON中包含\n"
                    "最终请输出以下 JSON 结构（包含所有人物的更新信息）：\n"
                    "{\n"
                    "  人物名1\": {\n"
                    "    \"events\": [...],\n"
                    "    \"utterances\": [...],\n"
                    "    \"speech_style\": \"...\",\n"
                    "    \"personality_traits\": [...],\n"
                    "    \"emotion_state\": \"...\",\n"
                    "    \"relations\": [...]\n"
                    "  },\n"
                    "  人物名2\": {\n"
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
                    "5. 【重要提醒】如某个人物在本章节中完全没有出现，强烈建议省略该人物，不在输出JSON中包含，这样可以显著提高处理效率。\n"
                    "6. 对于确实需要保留但无更新的人物，可以返回空的更新信息（所有字段为空）。\n"
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

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", model: str = "gpt-4o", backend: str = "openai", novel_name: str = "未知小说"):
        """初始化记忆构建器
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            backend: 后端类型
            novel_name: 小说名称，用于prompt命名
        """
        self.api_client = MemOSLLMClient(api_key, api_base, model, backend)
        self.memcubes = {}  # 全局人物记忆库
        self.alias_to_name = {}  # 别名到标准名映射
        self.novel_name = novel_name  # 小说名称
        self.enable_prompt_reuse = True  # 是否启用prompt复用
        self.skip_processed_chapters = True  # 是否跳过已处理的章节
        
        logger.info(f"小说记忆构建器初始化完成 - 小说: {novel_name}")
        logger.info(f"Prompt复用功能: {'启用' if self.enable_prompt_reuse else '禁用'}")
        logger.info(f"章节跳过功能: {'启用' if self.skip_processed_chapters else '禁用'}")

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
            "first_appearance_chapter": chunk_id,  # 添加章节级别的首次出现记录
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

    def load_memcubes_from_file(self, file_path: str):
        """从文件加载并更新memcubes"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            saved_result = data.get("saved_result")
            if saved_result:
                # saved_result是字典，需要解析其中的content字段
                content = saved_result.get("content")
                if content and isinstance(content, str):
                    # content是JSON字符串，需要解析
                    # 提取JSON部分（去掉```json和```标记）
                    if content.startswith("```json\n"):
                        content = content[8:]  # 去掉```json\n
                    if content.endswith("\n```"):
                        content = content[:-4]  # 去掉\n```
                    
                    character_data = json.loads(content)
                    
                    # 更新全局memcubes
                    for character_name, memcube in character_data.items():
                        if character_name in self.memcubes:
                            # 智能合并现有memcube，避免丢失历史数据
                            existing_memcube = self.memcubes[character_name]
                            
                            # 智能合并事件列表
                            existing_memcube["events"] = self.merge_events(
                                existing_memcube.get("events", []), 
                                memcube.get("events", [])
                            )
                            
                            # 合并话语列表（追加新的话语）
                            existing_memcube["utterances"].extend(memcube.get("utterances", []))
                            
                            # 更新语言风格（如果有新的）
                            if memcube.get("speech_style"):
                                existing_memcube["speech_style"] = memcube["speech_style"]
                            
                            # 合并性格特征（去重）
                            existing_memcube["personality_traits"] = self.merge_unique_list(
                                existing_memcube.get("personality_traits", []), 
                                memcube.get("personality_traits", [])
                            )
                            
                            # 更新情绪状态（如果有新的）
                            if memcube.get("emotion_state"):
                                existing_memcube["emotion_state"] = memcube["emotion_state"]
                            
                            # 合并关系列表（追加新的关系）
                            existing_memcube["relations"].extend(memcube.get("relations", []))
                            
                            # 合并别名列表（去重）
                            existing_memcube["aliases"] = self.merge_unique_list(
                                existing_memcube.get("aliases", []), 
                                memcube.get("aliases", [])
                            )
                            
                            # 保持最早的首次出现章节
                            if "first_appearance_chapter" in memcube and "first_appearance_chapter" not in existing_memcube:
                                existing_memcube["first_appearance_chapter"] = memcube["first_appearance_chapter"]
                            elif "first_appearance" in memcube and "first_appearance_chapter" not in existing_memcube:
                                existing_memcube["first_appearance_chapter"] = memcube["first_appearance"]
                            
                            logger.info(f"智能合并人物记忆: {character_name}")
                        else:
                            self.memcubes[character_name] = memcube
                            logger.info(f"新增人物记忆: {character_name}")
                    logger.info(f"从 {os.path.basename(file_path)} 加载了 {len(character_data)} 个人物记忆。")

        except FileNotFoundError:
            logger.warning(f"文件未找到: {file_path}")
        except json.JSONDecodeError:
            logger.error(f"JSON解码失败: {file_path}")
        except Exception as e:
            logger.error(f"加载memcubes时发生未知错误: {e}")

    def check_character_appears_in_text(self, character_name: str, text: str, threshold: float = 0.6) -> bool:
        """检查人物是否在文本中出现
        
        Args:
            character_name: 人物标准姓名
            text: 要检查的文本内容
            threshold: 匹配阈值，用于模糊匹配
            
        Returns:
            bool: 人物是否在文本中出现
        """
        try:
            # 导入字符串匹配库
            try:
                from difflib import SequenceMatcher
                from fuzzywuzzy import fuzz, process
                has_fuzzywuzzy = True
            except ImportError:
                from difflib import SequenceMatcher
                has_fuzzywuzzy = False
                logger.warning("⚠ fuzzywuzzy库未安装，将使用基础匹配策略")
            
            # 获取人物的所有别名
            character_aliases = []
            if character_name in self.memcubes:
                character_aliases = self.memcubes[character_name].get("aliases", [character_name])
            else:
                character_aliases = [character_name]
            
            # 添加从全局别名映射中找到的别名
            for alias, name in self.alias_to_name.items():
                if name == character_name and alias not in character_aliases:
                    character_aliases.append(alias)
            
            # 1. 精确匹配检查
            for alias in character_aliases:
                if alias in text:
                    logger.debug(f"✓ 人物 {character_name} 通过别名 '{alias}' 精确匹配")
                    return True
            
            # 2. 模糊匹配检查（使用fuzzywuzzy或difflib）
            text_words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)  # 提取中文词汇和英文单词
            
            for alias in character_aliases:
                if has_fuzzywuzzy:
                    # 使用fuzzywuzzy进行模糊匹配
                    best_match = process.extractOne(alias, text_words)
                    if best_match and best_match[1] >= threshold * 100:
                        logger.debug(f"✓ 人物 {character_name} 通过别名 '{alias}' 模糊匹配到 '{best_match[0]}' (相似度: {best_match[1]}%)")
                        return True
                else:
                    # 使用difflib进行基础模糊匹配
                    for word in text_words:
                        similarity = SequenceMatcher(None, alias, word).ratio()
                        if similarity >= threshold:
                            logger.debug(f"✓ 人物 {character_name} 通过别名 '{alias}' 模糊匹配到 '{word}' (相似度: {similarity:.2f})")
                            return True
            
            # 3. 部分匹配检查（对于复合姓名）
            for alias in character_aliases:
                if len(alias) >= 2:
                    # 检查姓氏或名字的部分匹配
                    for i in range(1, len(alias)):
                        partial_name = alias[:i]  # 前缀
                        if len(partial_name) >= 1 and partial_name in text:
                            logger.debug(f"✓ 人物 {character_name} 通过部分匹配 '{partial_name}' 找到")
                            return True
                        
                        partial_name = alias[i:]  # 后缀
                        if len(partial_name) >= 1 and partial_name in text:
                            logger.debug(f"✓ 人物 {character_name} 通过部分匹配 '{partial_name}' 找到")
                            return True
            
            logger.debug(f"✗ 人物 {character_name} 在文本中未找到")
            return False
            
        except Exception as e:
            logger.warning(f"检查人物出现时发生错误: {e}")
            # 发生错误时，默认返回True以避免误过滤
            return True

    def filter_active_characters(self, characters_data: dict, text: str) -> dict:
        """过滤在当前文本中出现的活跃人物 - 使用完全匹配策略
        
        Args:
            characters_data: 所有人物数据字典
            text: 当前章节文本
            
        Returns:
            dict: 过滤后的活跃人物数据
        """
        active_characters = {}
        total_characters = len(characters_data)
        
        logger.info(f"开始过滤活跃人物，总人物数: {total_characters}")
        
        for character_name, memcube in characters_data.items():
            # 获取人物的所有可能名称（包括别名）
            character_names = [character_name]
            
            # 从memcube中获取别名
            if isinstance(memcube, dict) and "aliases" in memcube:
                aliases = memcube["aliases"]
                if isinstance(aliases, list):
                    character_names.extend(aliases)
            
            # 从全局别名映射中获取别名
            for alias, name in self.alias_to_name.items():
                if name == character_name and alias not in character_names:
                    character_names.append(alias)
            
            # 检查是否有任何名称在文本中完全匹配
            found = False
            matched_name = None
            for name in character_names:
                if name and name in text:
                    found = True
                    matched_name = name
                    break
            
            if found:
                active_characters[character_name] = memcube
                logger.debug(f"✓ 活跃人物: {character_name} (匹配名称: {matched_name})")
            else:
                logger.debug(f"✗ 非活跃人物: {character_name} (检查名称: {character_names})")
        
        active_count = len(active_characters)
        filtered_count = total_characters - active_count
        
        logger.info(f"人物过滤结果: 活跃 {active_count} 人，过滤 {filtered_count} 人")
        
        # 如果过滤后人物太少，记录警告但仍返回过滤结果
        if active_count == 0:
            logger.warning("⚠ 过滤后无活跃人物，可能文本中没有提到任何已知人物")
            return {}
        elif active_count < total_characters * 0.3:  # 如果活跃人物少于30%
            logger.info(f"ℹ 活跃人物比例: {active_count}/{total_characters}={active_count/total_characters:.1%}")
        
        return active_characters

    def merge_events(self, old_events: list, new_events: list):
        """智能事件合并 - 处理状态更新和新增事件，支持跨章节事件延续"""
        event_dict = {e["event_id"]: e for e in old_events}

        for new_event in new_events:
            eid = new_event["event_id"]
            if eid in event_dict:
                # 延续事件的智能合并策略
                old_event = event_dict[eid]
                merged = old_event.copy()
                
                # 合并策略：保留历史信息，更新最新进展
                for key, value in new_event.items():
                    if key == "event":
                        # 事件描述：如果新描述更详细或包含更多信息，则使用新描述
                        if value and (len(value) > len(merged.get(key, "")) or "从第" in value):
                            merged[key] = value
                    elif key == "time":
                        # 时间：使用最新的时间描述
                        if value:
                            merged[key] = value
                    elif key == "if_completed":
                        # 完成状态：如果新状态为true，则更新
                        if value is True:
                            merged[key] = value
                    elif key in ["action", "motivation", "impact", "location"]:
                        # 其他字段：新值优先，但保留非空的历史值
                        if value not in [None, "", []]:
                            merged[key] = value
                    elif key == "involved_entities":
                        # 参与实体：合并去重
                        old_entities = set(merged.get(key, []))
                        new_entities = set(value or [])
                        merged[key] = list(old_entities | new_entities)
                
                event_dict[eid] = merged
                logger.info(f"🔄 事件延续更新: {eid}")
            else:
                # 全新事件直接加入
                event_dict[eid] = new_event
                logger.info(f"新事件添加: {eid}")

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

    def interactive_prompt_selection(self, chapter_id: str, task_type: TaskType, step_name: str) -> bool:
        """交互式prompt选择界面
        
        Args:
            chapter_id: 章节ID
            task_type: 任务类型
            step_name: 步骤名称
            
        Returns:
            bool: True表示使用已保存的prompt，False表示重新执行
        """
        # 创建正则表达式模式，避免在f-string中使用反斜杠
        pattern = r'[^\w\u4e00-\u9fff]'
        safe_novel_name = re.sub(pattern, '_', self.novel_name)
        safe_chapter_id = re.sub(pattern, '_', chapter_id)
        safe_step_name = re.sub(pattern, '_', step_name)
        prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{task_type.value}_{safe_step_name}"
        
        if not self.check_prompt_exists(chapter_id, task_type, step_name):
            logger.info(f"未找到已保存的prompt: {prompt_name}")
            return False
        
        print(f"\n🔍 发现已保存的prompt: {prompt_name}")
        print("选择操作:")
        print("1. 使用已保存的结果 (跳过API调用)")
        print("2. 重新执行 (忽略已保存的结果)")
        print("3. 查看已保存的prompt详情")
        
        while True:
            try:
                choice = input("请输入选择 (1/2/3): ").strip()
                
                if choice == "1":
                    logger.info(f"用户选择使用已保存的结果")
                    return True
                elif choice == "2":
                    logger.info(f"用户选择重新执行")
                    return False
                elif choice == "3":
                    self._show_prompt_details(chapter_id, task_type, step_name)
                    continue
                else:
                    print("无效选择，请输入 1、2 或 3")
                    
            except KeyboardInterrupt:
                print("\n 用户取消操作")
                return False
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                continue
    
    def _show_prompt_details(self, chapter_id: str, task_type: TaskType, step_name: str):
        """显示prompt详情"""
        # 创建正则表达式模式，避免在f-string中使用反斜杠
        pattern = r'[^\w\u4e00-\u9fff]'
        safe_novel_name = re.sub(pattern, '_', self.novel_name)
        safe_chapter_id = re.sub(pattern, '_', chapter_id)
        safe_step_name = re.sub(pattern, '_', step_name)
        prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{task_type.value}_{safe_step_name}"
        filepath = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            print(f"\n📋 Prompt详情: {prompt_name}")
            print(f"📅 创建时间: {prompt_data.get('timestamp', 'N/A')}")
            print(f"🔧 任务类型: {prompt_data.get('task_type', 'N/A')}")
            print(f"📖 小说名: {prompt_data.get('novel_name', 'N/A')}")
            print(f"📄 章节ID: {prompt_data.get('chapter_id', 'N/A')}")
            print(f"🔄 步骤名: {prompt_data.get('step_name', 'N/A')}")
            
            if 'saved_result' in prompt_data:
                result = prompt_data['saved_result']
                print(f"已保存结果状态: {result.get('status', 'N/A')}")
                print(f"结果保存时间: {prompt_data.get('result_saved_at', 'N/A')}")
                if result.get('status') == 'success':
                    content_preview = result.get('content', '')[:200]
                    print(f"内容预览: {content_preview}...")
            else:
                print("未找到已保存的结果")
                
            print("-" * 50)
            
        except Exception as e:
            print(f"读取prompt详情失败: {e}")

    def call_api_with_user_interaction(self, prompt: List[Dict], task_type: TaskType, 
                                     chapter_id: str, step_name: str, **kwargs) -> Dict:
        """带用户交互的API调用
        
        Args:
            prompt: 提示词消息列表
            task_type: 任务类型
            chapter_id: 章节ID
            step_name: 步骤名称
            **kwargs: 其他API调用参数
            
        Returns:
            Dict: API调用结果
        """
        # 如果启用了prompt复用，询问用户是否使用已保存的结果
        if self.enable_prompt_reuse:
            use_saved = self.interactive_prompt_selection(chapter_id, task_type, step_name)
            if use_saved:
                saved_result = self.get_saved_prompt_result(chapter_id, task_type, step_name)
                if saved_result:
                    return saved_result
        
        # 执行API调用
        return self.call_api_with_prompt_management(
            prompt, task_type, chapter_id, step_name, 
            force_new=True, **kwargs
        )

    def update_memcubes_batch(self, characters_data: dict, content: str, chapter_id: str):
        """批量更新多个人物的记忆立方体"""
        try:
            # 智能过滤：只处理在当前章节中出现的活跃人物
            logger.info(f"开始智能过滤人物，原始人物数: {len(characters_data)}")
            active_characters = self.filter_active_characters(characters_data, content)
            
            # 如果没有活跃人物，返回空结果
            if not active_characters:
                logger.warning("⚠ 没有检测到活跃人物，跳过批量更新")
                return {}, None
            
            # 构建批量更新提示（仅针对活跃人物）
            update_prompt = Prompt.update_characters_batch_prompt(active_characters, content)
            
            # 使用prompt管理功能调用API
            result = self.call_api_with_prompt_management(
                update_prompt, 
                TaskType.CHARACTER_ANALYSIS, 
                chapter_id, 
                "batch_update",
                timeout=1800
            )
            
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
            
            # 检查是否跳过已处理的章节
            if self.skip_processed_chapters:
                # 检查关键步骤是否已完成
                if (self.check_prompt_exists(chapter_id, TaskType.EVENT_EXTRACTION, "character_identification") and
                    self.check_prompt_exists(chapter_id, TaskType.CHARACTER_ANALYSIS, "batch_update")):
                    logger.info(f"⏭️ 跳过已处理的章节: {chapter_id}")
                    # 加载已保存的结果
                    # 构建prompt文件路径
                    pattern = r'[^\w\u4e00-\u9fff]'
                    safe_novel_name = re.sub(pattern, '_', self.novel_name)
                    safe_chapter_id = re.sub(pattern, '_', chapter_id)
                    safe_step_name = re.sub(pattern, '_', "batch_update")
                    prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{TaskType.CHARACTER_ANALYSIS.value}_{safe_step_name}"
                    prompt_path = os.path.join("prompts", f"{prompt_name}.json")
                    self.load_memcubes_from_file(prompt_path)
                    continue
      
            with open(chapter_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 执行人物识别和初始化
            self._process_character_identification(content, chapter_id)
            
            # 更新所有人物状态
            self._update_all_characters(content, chapter_id)

    def _process_character_identification(self, content: str, chapter_id: str):
        """处理人物识别"""
        name_prompt = Prompt.extract_character_names_prompt(content, self.alias_to_name)
        
        # 使用prompt管理功能调用API
        name_result = self.call_api_with_prompt_management(
            name_prompt, 
            TaskType.EVENT_EXTRACTION, 
            chapter_id, 
            "character_identification",
            timeout=1800
        )

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
        
        # 保存当前章节的人物和别名信息到JSON文件
        self._save_character_data_to_json(chapter_id)

    def _save_character_data_to_json(self, chapter_id: str):
        """保存当前章节处理后的人物和别名信息到JSON文件"""
        try:
            # 构建保存数据
            character_data = {
                "chapter_id": chapter_id,
                "timestamp": datetime.now().isoformat(),
                "characters": {},
                "alias_mapping": dict(self.alias_to_name)
            }
            
            # 收集所有人物信息
            for std_name, memcube in self.memcubes.items():
                character_data["characters"][std_name] = {
                    "name": std_name,
                    "aliases": memcube.get("aliases", []),
                    "first_appearance_chapter": memcube.get("first_appearance_chapter", memcube.get("first_appearance", chapter_id)),
                    "total_aliases_count": len(memcube.get("aliases", []))
                }
            
            # 创建输出目录
            output_dir = "character_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存到JSON文件
            filename = f"{output_dir}/characters_{chapter_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(character_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"人物数据已保存到: {filename}")
            logger.info(f"本章节共识别 {len(character_data['characters'])} 个人物")
            
        except Exception as e:
            logger.error(f"保存人物数据失败: {str(e)}")

    def _update_all_characters(self, content: str, chapter_id: str):
        """批量更新所有人物状态"""
        if not self.memcubes:
            logger.info("没有人物需要更新")
            return
            
        try:
            # 批量调用API更新所有人物
            updated_data, error = self.update_memcubes_batch(self.memcubes, content, chapter_id)
            
            if error:
                logger.warning(f"批量更新失败 in {chapter_id} -> {error}")
                return
                
            if not updated_data:
                logger.warning(f"批量更新返回空结果 in {chapter_id}")
                return

            # 处理批量更新结果
            for name, updated in updated_data.items():
                if name not in self.memcubes:
                    logger.warning(f"返回了未知人物：{name}")
                    continue
                    
                if not updated:
                    logger.info(f"{name} 在本章节中无更新")
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
                
                logger.info(f"成功更新：{name}")
            logger.info("更新完毕")

        except Exception as e:
            logger.error(f"批量更新异常 in {chapter_id} -> {e}")

    def save_memcubes(self, filename: str = "memcubes.json"):
        """保存记忆立方体"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.memcubes, f, ensure_ascii=False, indent=2)
        logger.info(f"记忆立方体已保存到 {filename}")

    def load_memcubes(self, filename: str = "memcubes.json"):
        """加载记忆立方体"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.memcubes = json.load(f)
            logger.info(f"记忆立方体已从 {filename} 加载")
        except FileNotFoundError:
            logger.warning(f"文件 {filename} 不存在")

    def check_prompt_exists(self, chapter_id: str, task_type: TaskType, step_name: str) -> bool:
        """检查指定的prompt是否已存在
        
        Args:
            chapter_id: 章节ID
            task_type: 任务类型
            step_name: 步骤名称
            
        Returns:
            bool: prompt是否存在
        """
        # 创建正则表达式模式，避免在f-string中使用反斜杠
        pattern = r'[^\w\u4e00-\u9fff]'
        safe_novel_name = re.sub(pattern, '_', self.novel_name)
        safe_chapter_id = re.sub(pattern, '_', chapter_id)
        safe_step_name = re.sub(pattern, '_', step_name)
        prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{task_type.value}_{safe_step_name}"
        filepath = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
        return os.path.exists(filepath)
    
    def get_saved_prompt_result(self, chapter_id: str, task_type: TaskType, step_name: str) -> Optional[Dict]:
        """获取已保存的prompt结果
        
        Args:
            chapter_id: 章节ID
            task_type: 任务类型
            step_name: 步骤名称
            
        Returns:
            Optional[Dict]: 保存的结果数据，如果不存在则返回None
        """
        # 创建正则表达式模式，避免在f-string中使用反斜杠
        pattern = r'[^\w\u4e00-\u9fff]'
        safe_novel_name = re.sub(pattern, '_', self.novel_name)
        safe_chapter_id = re.sub(pattern, '_', chapter_id)
        safe_step_name = re.sub(pattern, '_', step_name)
        prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{task_type.value}_{safe_step_name}"
        filepath = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                # 检查是否有保存的结果
                if 'saved_result' in prompt_data:
                    logger.info(f"🔄 使用已保存的结果: {prompt_name}")
                    return prompt_data['saved_result']
        except Exception as e:
            logger.warning(f"读取已保存结果失败: {e}")
        
        return None
    
    def save_prompt_result(self, chapter_id: str, task_type: TaskType, step_name: str, result: Dict):
        """保存prompt执行结果到文件
        
        Args:
            chapter_id: 章节ID
            task_type: 任务类型
            step_name: 步骤名称
            result: 执行结果
        """
        # 创建正则表达式模式，避免在f-string中使用反斜杠
        pattern = r'[^\w\u4e00-\u9fff]'
        safe_novel_name = re.sub(pattern, '_', self.novel_name)
        safe_chapter_id = re.sub(pattern, '_', chapter_id)
        safe_step_name = re.sub(pattern, '_', step_name)
        prompt_name = f"{safe_novel_name}_{safe_chapter_id}_{task_type.value}_{safe_step_name}"
        filepath = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
        
        try:
            # 读取现有数据
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
            else:
                prompt_data = {}
            
            # 添加结果数据
            prompt_data['saved_result'] = result
            prompt_data['result_saved_at'] = datetime.now().isoformat()
            
            # 保存回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"结果已保存: {prompt_name}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def call_api_with_prompt_management(self, prompt: List[Dict], task_type: TaskType, 
                                      chapter_id: str, step_name: str, 
                                      force_new: bool = False, **kwargs) -> Dict:
        """带prompt管理的API调用
        
        Args:
            prompt: 提示词消息列表
            task_type: 任务类型
            chapter_id: 章节ID
            step_name: 步骤名称
            force_new: 是否强制重新调用（忽略已保存的结果）
            **kwargs: 其他API调用参数
            
        Returns:
            Dict: API调用结果
        """
        # 检查是否启用复用且不强制重新调用
        if self.enable_prompt_reuse and not force_new:
            # 先检查是否有已保存的结果
            saved_result = self.get_saved_prompt_result(chapter_id, task_type, step_name)
            if saved_result:
                logger.info(f"🔄 使用已保存的结果，跳过API调用: {chapter_id}_{step_name}")
                return saved_result
        
        # 调用API（会自动保存prompt）
        result = self.api_client.call_api(
            prompt, task_type, 
            novel_name=self.novel_name,
            chapter_id=chapter_id,
            step_name=step_name,
            **kwargs
        )
        
        # 保存结果
        if result.get("status") == "success":
            self.save_prompt_result(chapter_id, task_type, step_name, result)
        
        return result

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

        logger.info(f"完成转换，共生成 {len(nodes)} 个 memory 节点，{len(edges)} 条边")
        logger.info(f"输出文件: {output_file}")

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
            logger.info("武侠文字游戏初始化完成")
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
    """主函数 - 演示完整的使用流程，包含prompt复用功能"""
    # 配置参数
    API_KEY = "sk-7b3b452751e94c05bad166250033cd54"  # 请替换为你的API密钥
    API_BASE = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-reasoner"
    back_end = "deepseek"
    
    # 小说名称
    novel_name = "天龙八部"

    print("🚀 小说记忆构建器 - 支持Prompt复用功能")
    print("=" * 50)
    
    # 询问用户是否启用prompt复用功能
    print("\n⚙️ 配置选项:")
    print("1. 启用Prompt复用 (推荐) - 自动保存和复用API调用结果")
    print("2. 禁用Prompt复用 - 每次都重新调用API")
    
    while True:
        try:
            # choice = input("请选择 (1/2): ").strip()
            choice = "1"
            if choice == "1":
                enable_prompt_reuse = True
                print("已启用Prompt复用功能")
                break
            elif choice == "2":
                enable_prompt_reuse = False
                print("已禁用Prompt复用功能")
                break
            else:
                print("无效选择，请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return

    # 创建记忆构建器，传入小说名称
    builder = NovelMemoryBuilder(API_KEY, API_BASE, MODEL_NAME, back_end, novel_name=novel_name)
    
    # 设置prompt复用选项
    builder.enable_prompt_reuse = enable_prompt_reuse
    builder.skip_processed_chapters = enable_prompt_reuse  # 如果启用复用，也启用章节跳过

    # 如果有小说文件，提取章节
    novel_file = f"{novel_name}.txt"
    # if os.path.exists(novel_file):
    #     print(f"\n发现小说文件: {novel_file}")
    #     extract_choice = input("是否重新提取章节? (y/n): ").strip().lower()
    #     if extract_choice == 'y':
    #         logger.info("开始提取小说章节...")
    #         with open(novel_file, "r", encoding="utf-8") as f:
    #             full_text = f.read()
    #         builder.extract_all_chapters(full_text)
    #         print("章节提取完成")

    # 处理章节，构建记忆
    if os.path.exists("chapters"):
        print(f"\n开始构建人物记忆...")
        print(f"Prompt复用状态: {'启用' if enable_prompt_reuse else '禁用'}")
        print(f"章节跳过状态: {'启用' if builder.skip_processed_chapters else '禁用'}")
        
        if enable_prompt_reuse:
            print("\n提示: 处理过程中如果发现已保存的prompt，系统会询问您是否复用")
        
        builder.process_chapters()

        # 保存结果
        output_file = f"memcubes_{novel_name}_complete.json"
        builder.save_memcubes(output_file)
        print(f"记忆数据已保存到: {output_file}")

    # 演示情节推演
    if builder.memcubes and "段誉" in builder.memcubes:
        print("\n演示情节推演功能...")
        character_name = "段誉"
        user_input = "如果段誉没有出现在剑湖宫比武大会，事情会怎样？"

        prompt = Prompt.speculate_event_outcome(character_name, builder.memcubes[character_name], user_input)
        
        if enable_prompt_reuse:
            # 使用带用户交互的API调用
            response = builder.call_api_with_user_interaction(
                prompt, TaskType.PLOT_SPECULATION, "demo", "plot_speculation"
            )
        else:
            # 直接调用API
            response = builder.api_client.call_api(prompt, TaskType.PLOT_SPECULATION)
            
        print(f"\n🎬 情节推演结果：\n{response.get('content', '没有返回')}")
    
    print(f"\n🎉 程序执行完成！")
    if enable_prompt_reuse:
        print(f"Prompt文件保存在: {PROMPTS_DIR}")
        print("下次运行时可以复用这些结果，节省时间和成本")

if __name__ == "__main__":
    main()