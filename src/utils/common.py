"""
通用工具函数
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, file_path: str, indent: int = 2):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: str) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def ensure_dir(directory: str):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)


def decode_value(value: Any, mapping: Dict, default: str = "unknown") -> str:
    """根据映射表解码值"""
    if pd.isna(value):
        return default
    return mapping.get(int(value), default)


def reverse_likert(values, max_value: int = 5):
    """
    反转Likert量表分数

    Args:
        values: 单个值或Series
        max_value: 最大值（默认5）

    Returns:
        反转后的值或Series
    """
    if isinstance(values, pd.Series):
        return (max_value + 1) - values
    else:
        if pd.isna(values):
            return values
        return (max_value + 1) - values


def parse_multi_choice(value: str, separator: str = ',') -> List[str]:
    """解析多选题"""
    if pd.isna(value) or value == '':
        return []
    return [choice.strip() for choice in str(value).split(separator)]


def generate_unique_id(country: str, index: int, width: int = 4) -> str:
    """生成唯一ID"""
    return f"{country}_{str(index).zfill(width)}"


def calculate_income_level(income_code: int, config: Dict) -> str:
    """计算收入水平"""
    income_mapping = config['encodings']['income_level']
    for level, codes in income_mapping.items():
        if income_code in codes:
            return level
    return "unknown"


class QuestionMapper:
    """问卷字段映射器"""

    # 问卷列名到简化代码的映射
    COLUMN_MAPPING = {
        '1、您早晨通勤时，从家里到工作地点一般需要多长时间（以分钟为单位）？': 'Q1',
        '2、您下午通勤时，从工作地点到家里一般需要多长时间（以分钟为单位）？': 'Q2',
        '3、您通勤的主要路线是什么类型的道路？': 'Q3',
        '4、在一个典型的早晨通勤行程中（从家里到工作地点），您会在拥堵中花费多长时间（以分钟计）？': 'Q4',
        '5、在一个典型的下午通勤行程中（从工作地点到家里），您会在拥堵中花费多长时间（以分钟计）？': 'Q5',
        '6、您在何种程度上同意或不同意以下描述？—在我通常的通勤过程中，拥堵是一个问题。': 'Q6',
        '7、我在通勤时经常因为交通拥堵而改道其他路线。': 'Q7',
        '8、在改道其他路线时，我很相信自己正在做出最好的选择。': 'Q8',
        '9、改道其他路线后，我有可能在通过拥堵地区后返回正常路线。': 'Q9',
        '10、如何获取交通状况或路线指引的信息？': 'Q10',
        '11、在出行过程中，您什么时候会寻求交通状况和路线信息？': 'Q11',
        '12、您在何种程度上同意或不同意以下描述？—我经常寻求交通信息/路线引导。': 'Q12',
        '13、在获取交通信息时，我经常不同意所得到的路线引导。': 'Q13',
        '14、获取交通信息时，我通常不理会路线引导，并转而自己选择路线。': 'Q14',
        '15、我经常因为交通拥堵的信息而取消行程或改变目的地。': 'Q15',
        '16、在以下每种情景下，您在何种程度上同意或不同意以下说法？- 遇到以下情景时，我会从正常（常走）路线改道行驶。—作业区（施工、维修、作业团队等）。': 'Q16',
        '17、特殊事件（音乐会、体育比赛、节假日交通等）': 'Q17',
        '18、天气事件（由雨、雪、雾导致的行驶速度下降）': 'Q18',
        '19、高峰期（车流量过大导致的交通瓶颈）': 'Q19',
        '20、事故（交通事故、抛锚车辆、道路上出现障碍物等）': 'Q20',
        '21、在您的通勤中，多长时间的预期延误会导致您寻找另一条路线（以分钟为单位）？': 'Q21',
        '22、您在何种程度上同意或不同意以下描述？—我更倾向于在我熟悉的路线上行驶。': 'Q22',
        '23、如果我对该地区的环境比较熟悉，那么我会更愿意改道其他路线。': 'Q23',
        '24、相比于地方街道，我更倾向于改道到高速公路。': 'Q24',
        '25、旅行时间的一致性/可靠性对我和我的路线选择很重要。': 'Q25',
        '26、相比于只有一种改道路线，如果有多种改道选择，我会更愿意改道。': 'Q26',
        '27、如果改道路线比继续在正常路线上行驶需要更长的时间，我就不会改道。': 'Q27',
        '28、因事故或作业区而关闭的车道数量很大程度上影响了我的改道意愿。': 'Q28',
        '29、即使可能需要更长的时间，我也会选择一条顺畅的路线而不是一条拥堵的路线。': 'Q29',
        '30、只要我认为可以节省时间，我就会立即改道。': 'Q30',
        '31、请对这个陈述选择 "有点不同意"。': 'Q31',
        '32、替代路线上的停车标志/交通信号灯的数量对我选择该路线的意愿有很大影响。': 'Q32',
        '33、如果我离目的地很远，我会更愿意改变路线。': 'Q33',
        '34、我避免开车经过有作业区的地区。': 'Q34',
        '35、我知道计划中的特殊事件（如体育赛事、交通管制等）可能会影响我通勤时的交通状况。': 'Q35',
        '36、我认为带有旅行时间/事故信息的"可变信息板"是有用且准确的。(下面提供了"可变信息板"的图片)': 'Q36',
        '37、如果有"可变信息板"的建议，我更愿意改变路线。(下面提供"可变信息板"的图片)': 'Q37',
        '38、如果我看到其他驾驶员也在改道其他路线，我也就更愿意改道。': 'Q38',
        '39、车上载有乘客时，我的驾驶方式会有所不同。': 'Q39',
        '40、我认为我是一个激进的驾驶员。': 'Q40',
        '41、我认为我是一个谨慎的驾驶员。': 'Q41',
        '42、您在交通拥堵中等待多长时间后才会决定改道其他路线（以分钟计）？': 'Q42',
        '43、对于以下出行类型，您在何种程度上同意或不同意以下说法？\n\n- 在以下出行类型中，我可能会改道其他路线。—通勤（早晨；从家里到工作地点）': 'Q43',
        '44、通勤（下午；从工作地点到家里）': 'Q44',
        '45、购物出行': 'Q45',
        '46、短途的休闲社交出行（如去公园、访问朋友、喝咖啡等）': 'Q46',
        '47、计划好的长途度假出行（如去其他城市、国家森林公园等）': 'Q47',
        '48、紧急疏散': 'Q48',
        '49、您的性别是？': 'Q49',
        '50、您的年龄是？': 'Q50',
        '51、您的民族是？': 'Q51',
        '52、您的家庭年收入是多少？': 'Q52',
        '53、您的职业最符合以下哪项描述？': 'Q53',
        '32、您的最高教育程度是？': 'Q54',  # 注意：字段名是32但实际是Q54
        '55、您如何描述您所驾驶的车辆？': 'Q55',
        '56、您驾驶的车辆是否配备了导航系统？': 'Q56',
        '57、您如何描述您的住所（家）的位置？': 'Q57',
        '58、您如何描述您的工作场所的位置？': 'Q58',
    }

    @classmethod
    def rename_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """重命名DataFrame列为简化代码"""
        rename_dict = {k: v for k, v in cls.COLUMN_MAPPING.items() if k in df.columns}
        return df.rename(columns=rename_dict)

    @classmethod
    def get_question_text(cls, question_code: str) -> str:
        """根据问题代码获取原始问题文本"""
        reverse_mapping = {v: k for k, v in cls.COLUMN_MAPPING.items()}
        return reverse_mapping.get(question_code, question_code)
