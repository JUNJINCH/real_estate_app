from model.train_model import train_and_evaluate, plot_feature_importance
from model.save_model import save_model
import subprocess
import webbrowser
import time

if __name__ == "__main__":
    model, X_train = train_and_evaluate()
    save_model(model)
    plot_feature_importance(model, X_train)

    

print("\n Training complete. Launching Streamlit app...")

# 启动 Streamlit 应用（非阻塞方式）
subprocess.Popen(["streamlit", "run", "app/streamlit_app.py"])

# 等待几秒打开网页
time.sleep(2)
webbrowser.open("http://localhost:8501")