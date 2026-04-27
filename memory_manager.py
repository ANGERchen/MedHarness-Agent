import sqlite3
import hashlib
import json

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

class HealthMemory:
    def __init__(self, db_path="health_system.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles 
            (user_id TEXT PRIMARY KEY, password_hash TEXT, age INTEGER, weight REAL, 
             goals TEXT, medical_history TEXT, long_term_tags TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS health_metrics 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, date TEXT, 
             steps INTEGER, sleep_hours REAL)''')
             
        # 新增：用于 AutoHarness 评估的交互日志表
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_logs
            (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, query TEXT, 
             context TEXT, response TEXT, eval_score REAL, eval_reason TEXT)''')
        self.conn.commit()

    def register_user(self, user_id, password):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,))
        if cursor.fetchone(): return False, "用户 ID 已存在"
        
        pwd_hash = hash_password(password)
        cursor.execute("INSERT INTO user_profiles (user_id, password_hash, age, weight, goals, medical_history, long_term_tags) VALUES (?,?,?,?,?,?,?)", 
                       (user_id, pwd_hash, 25, 65.0, "通用健康", "无", "{}"))
        self.conn.commit()
        return True, "注册成功"

    def verify_user(self, user_id, password):
        cursor = self.conn.cursor()
        cursor.execute("SELECT password_hash FROM user_profiles WHERE user_id=?", (user_id,))
        row = cursor.fetchone()
        if not row: return False, "用户不存在"
        if row[0] == hash_password(password): return True, "登录成功"
        return False, "密码错误"

    def get_profile(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT age, weight, goals, medical_history, long_term_tags FROM user_profiles WHERE user_id=?", (user_id,))
        row = cursor.fetchone()
        if row: return {"age": row[0], "weight": row[1], "goals": row[2], "history": row[3], "tags": json.loads(row[4] or "{}")}
        return {}

    def update_long_term_tags(self, user_id, new_tags: dict):
        """旁路记忆更新接口：固化大模型提取的用户特征"""
        current_tags = self.get_profile(user_id).get("tags", {})
        current_tags.update(new_tags)
        cursor = self.conn.cursor()
        cursor.execute("UPDATE user_profiles SET long_term_tags=? WHERE user_id=?", (json.dumps(current_tags), user_id))
        self.conn.commit()

    def log_chat(self, user_id, query, context, response):
        """记录对话，供后续 Harness 离线评估"""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO chat_logs (user_id, query, context, response) VALUES (?,?,?,?)",
                       (user_id, query, context, response))
        self.conn.commit()
    def update_metrics(self, user_id, date, steps, sleep):
        """补充：写入每日健康指标"""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO health_metrics (user_id, date, steps, sleep_hours) VALUES (?,?,?,?)", 
                       (user_id, date, steps, sleep))
        self.conn.commit()

    def get_latest_metrics(self, user_id):
        """补充：读取最新健康指标"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT steps, sleep_hours FROM health_metrics WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,))
        return cursor.fetchone() or (0, 0.0) 
memory = HealthMemory()