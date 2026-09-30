
import os, sqlite3, time
from datetime import datetime
import cv2, numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

app=Flask(__name__)
UPLOAD_FOLDER="static/uploads"
MODEL_PATH="model/blur_detector.h5"
DB_PATH="history.db"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

try:
    from tensorflow.keras.models import load_model
    model=load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
except Exception:
    model=None

def db():
    con=sqlite3.connect(DB_PATH)
    con.row_factory=sqlite3.Row
    return con

def init_db():
    con=db()
    con.execute("""CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT, prediction TEXT, confidence REAL,
        blur_score REAL, width INTEGER, height INTEGER,
        file_size REAL, processing_ms REAL, created_at TEXT)""")
    con.commit(); con.close()

def blur_score(path):
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray,cv2.CV_64F).var()), img

def predict(path):
    score,img=blur_score(path)
    start=time.perf_counter()
    if model is not None:
        x=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        x=cv2.resize(x,(128,128)).astype("float32")/255.0
        p=float(model.predict(np.expand_dims(x,0),verbose=0)[0][0])
        label="Blurred" if p>=0.5 else "Clear"
        confidence=(p if label=="Blurred" else 1-p)*100
    else:
        # Demonstration fallback if no trained model exists.
        threshold=100.0
        label="Blurred" if score<threshold else "Clear"
        confidence=min(99.0,max(50.0,50.0+abs(score-threshold)/max(threshold,1)*50))
    ms=(time.perf_counter()-start)*1000
    return label,confidence,score,img,ms

@app.route("/")
def dashboard():
    con=db()
    rows=con.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()
    total=len(rows); blurred=sum(r["prediction"]=="Blurred" for r in rows)
    clear=total-blurred
    avg=round(sum(r["confidence"] for r in rows)/total,2) if total else 0
    con.close()
    return render_template("dashboard.html",active="dashboard",
                           total=total,blurred=blurred,clear=clear,avg=avg)

@app.route("/analyze",methods=["GET","POST"])
def analyze():
    result=None; error=None
    if request.method=="POST":
        f=request.files.get("image")
        if not f or not f.filename:
            error="Please select an image."
        else:
            ext=os.path.splitext(f.filename)[1].lower()
            if ext not in [".jpg",".jpeg",".png",".bmp",".webp"]:
                error="Supported formats: JPG, JPEG, PNG, BMP, WEBP."
            else:
                name=secure_filename(f.filename)
                stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename=stamp+"_"+name
                path=os.path.join(UPLOAD_FOLDER,filename)
                f.save(path)
                try:
                    label,conf,score,img,ms=predict(path)
                    h,w=img.shape[:2]
                    size=os.path.getsize(path)/1024
                    con=db()
                    con.execute("""INSERT INTO scans
                    (filename,prediction,confidence,blur_score,width,height,file_size,processing_ms,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (filename,label,conf,score,w,h,size,ms,datetime.now().strftime("%d %b %Y, %I:%M %p")))
                    con.commit(); con.close()
                    result=dict(filename=filename,prediction=label,confidence=round(conf,2),
                                blur_score=round(score,2),width=w,height=h,
                                file_size=round(size,2),processing_ms=round(ms,2))
                except Exception as e:
                    error="Could not analyze the image: "+str(e)
    return render_template("analyze.html",active="analyze",result=result,error=error)

@app.route("/analytics")
def analytics():
    con=db()
    rows=con.execute("SELECT * FROM scans ORDER BY id ASC").fetchall()
    con.close()
    clear=sum(r["prediction"]=="Clear" for r in rows)
    blurred=sum(r["prediction"]=="Blurred" for r in rows)
    avg=round(sum(r["confidence"] for r in rows)/len(rows),2) if rows else 0
    daily={}
    for r in rows:
        day=r["created_at"].split(",")[0]
        daily[day]=daily.get(day,0)+1
    return render_template("analytics.html",active="analytics",
        clear=clear,blurred=blurred,avg=avg,
        labels=list(daily.keys()),values=list(daily.values()))

@app.route("/history")
def history():
    con=db()
    rows=con.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()
    con.close()
    return render_template("history.html",active="history",rows=rows)

@app.route("/api/stats")
def stats():
    con=db()
    rows=con.execute("SELECT * FROM scans").fetchall()
    con.close()
    total=len(rows); blurred=sum(r["prediction"]=="Blurred" for r in rows)
    return jsonify({"total":total,"clear":total-blurred,"blurred":blurred,
                    "average_confidence":round(sum(r["confidence"] for r in rows)/total,2) if total else 0})

@app.route("/clear-history",methods=["POST"])
def clear_history():
    con=db(); con.execute("DELETE FROM scans"); con.commit(); con.close()
    return redirect(url_for("history"))

@app.route("/about")
def about():
    return render_template("about.html",active="about")

@app.route("/settings")
def settings():
    return render_template("settings.html",active="settings")

if __name__=="__main__":
    init_db()
    app.run(debug=True)
