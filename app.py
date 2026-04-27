import streamlit as st
import os
from agent_core import run_health_agent
from memory_manager import memory

st.set_page_config(page_title="AI 架构师健康管家", layout="wide")

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "current_user" not in st.session_state: st.session_state.current_user = ""

if not st.session_state.logged_in:
    st.title("🏥 智能体登录台")
    uid = st.text_input("用户名")
    pwd = st.text_input("密码", type="password")
    if st.button("登录/注册"):
        success, msg = memory.register_user(uid, pwd)
        if not success: _, msg = memory.verify_user(uid, pwd)
        if "成功" in msg:
            st.session_state.logged_in = True
            st.session_state.current_user = uid
            st.rerun()
        else: st.error(msg)
else:
    uid = st.session_state.current_user
    with st.sidebar:
        st.title(f"👤 {uid}")
        profile = memory.get_profile(uid)
        st.json(profile.get("tags", {})) # 展示长程记忆提取出的动态标签
        st.divider()
        st.subheader("📥 每日指标同步")
        with st.form("data_form"):
            import datetime
            steps = st.number_input("今日步数", value=8000, step=100)
            sleep = st.number_input("昨晚睡眠 (小时)", value=7.5, step=0.5)
            if st.form_submit_button("同步数据"):
                memory.update_metrics(uid, str(datetime.date.today()), steps, sleep)
                st.success("数据已保存！")
        if st.button("退出"):
            st.session_state.logged_in = False
            st.rerun()
    col_chat, col_dash = st.columns([2, 1])
    with col_dash:
        st.subheader("📊 实时仪表盘")
        latest_steps, latest_sleep = memory.get_latest_metrics(uid)
        c1, c2 = st.columns(2)
        c1.metric("今日步数", f"{latest_steps}")
        c2.metric("睡眠时间", f"{latest_sleep}h")
        st.divider()
        st.info("💡 小贴士：保持数据同步可让 AI 建议更精准。")
    with col_chat:
        st.subheader("💬 对话")
        if "chat_hist" not in st.session_state: st.session_state.chat_hist = []
    
    # 渲染历史对话
        for m in st.session_state.chat_hist:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
    # ================= 修复点：图片上传组件 =================
        uploaded_file = st.file_uploader("📷 附加图片 (如食物照片、体检报告，非必填)", type=["jpg","png","jpeg"])
    
        if prompt := st.chat_input("输入健康问题..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_hist.append({"role": "user", "content": prompt})
        
        # 将用户上传的图片暂存到本地，供视觉模型读取
            img_path = None
            if uploaded_file:
                img_path = f"temp_{uid}.jpg" # 根据用户ID隔离临时文件
                with open(img_path, "wb") as f: 
                    f.write(uploaded_file.getbuffer())
        
            with st.chat_message("assistant"):
                with st.spinner("主控 Agent 正在思考与调度子系统..."):
                    try:
                    # 将图片路径传入底层引擎，激活 image_analysis 路由
                        reply = run_health_agent(prompt, uid, image_path=img_path, chat_history=st.session_state.chat_hist)
                        st.markdown(reply)
                        st.session_state.chat_hist.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"调度异常：{str(e)}")
                
        # 阅后即焚：推理完成后自动清理本地图片，释放服务器空间
            if img_path and os.path.exists(img_path):
                os.remove(img_path)