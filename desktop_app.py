import threading
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

from model import predict_one
from train_utils import get_saved_or_train_model


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Job Role Finder")
        self.root.geometry("760x600")
        self.root.resizable(False, False)

        self.model = None

        self.education_list = ["Bachelor", "College", "Master", "Bootcamp", "Certification", "Unknown"]
        self.language_list = ["English", "Russian", "Kazakh", "Turkish", "German"]
        self.cert_list = [
            "None", "AWS Cloud Practitioner", "Google Data Analytics", "Cisco CCNA",
            "Oracle Java", "Meta Frontend", "TensorFlow Developer", "ISTQB Foundation",
            "Kubernetes Basics", "Unknown",
        ]

        self.make_window()
        threading.Thread(target=self.load_model, daemon=True).start()

    def make_window(self):
        self.root.configure(bg="#f3f4f6")

        box = ttk.Frame(self.root, padding=20)
        box.pack(fill="both", expand=True, padx=18, pady=18)

        ttk.Label(box, text="Job Role Finder", font=("Segoe UI", 20, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(box, text="Fill candidate details below. The program will suggest a possible IT role.").grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 18))

        ttk.Label(box, text="Skills").grid(row=2, column=0, columnspan=2, sticky="w")
        self.skills = tk.Text(box, height=4, width=78)
        self.skills.grid(row=3, column=0, columnspan=2, sticky="we", pady=(4, 12))
        self.skills.insert("1.0", "Python, FastAPI, REST API, PostgreSQL, Docker, SQL")

        ttk.Label(box, text="Qualification / project").grid(row=4, column=0, sticky="w")
        self.qualification = ttk.Entry(box, width=38)
        self.qualification.grid(row=5, column=0, sticky="w", pady=(4, 12))
        self.qualification.insert(0, "Backend project")

        ttk.Label(box, text="Experience years").grid(row=4, column=1, sticky="w", padx=(20, 0))
        self.exp = ttk.Spinbox(box, from_=0, to=20, width=14)
        self.exp.grid(row=5, column=1, sticky="w", padx=(20, 0), pady=(4, 12))
        self.exp.set(1)

        ttk.Label(box, text="Education").grid(row=6, column=0, sticky="w")
        self.education = ttk.Combobox(box, values=self.education_list, state="readonly", width=35)
        self.education.grid(row=7, column=0, sticky="w", pady=(4, 12))
        self.education.set("Bachelor")

        ttk.Label(box, text="Language").grid(row=6, column=1, sticky="w", padx=(20, 0))
        self.language = ttk.Combobox(box, values=self.language_list, state="readonly", width=28)
        self.language.grid(row=7, column=1, sticky="w", padx=(20, 0), pady=(4, 12))
        self.language.set("English")

        ttk.Label(box, text="Certificate").grid(row=8, column=0, sticky="w")
        self.cert = ttk.Combobox(box, values=self.cert_list, state="readonly", width=35)
        self.cert.grid(row=9, column=0, sticky="w", pady=(4, 12))
        self.cert.set("None")

        ttk.Label(box, text="Soft skills").grid(row=8, column=1, sticky="w", padx=(20, 0))
        self.soft = ttk.Entry(box, width=32)
        self.soft.grid(row=9, column=1, sticky="w", padx=(20, 0), pady=(4, 12))
        self.soft.insert(0, "Communication, Problem Solving")

        self.status = ttk.Label(box, text="Preparing model...")
        self.status.grid(row=10, column=0, sticky="w", pady=(2, 12))

        self.btn = ttk.Button(box, text="Check role", command=self.predict)
        self.btn.grid(row=10, column=1, sticky="e", pady=(2, 12))
        self.btn.config(state="disabled")

        result_frame = ttk.LabelFrame(box, text="Output", padding=12)
        result_frame.grid(row=11, column=0, columnspan=2, sticky="we", pady=(8, 0))

        self.result = ttk.Label(result_frame, text="No prediction yet.", font=("Segoe UI", 14, "bold"))
        self.result.pack(anchor="w")

        self.details = tk.Text(result_frame, height=6, width=82)
        self.details.pack(fill="x", pady=(10, 0))
        self.details.insert("1.0", "Top roles will be shown here.")
        self.details.config(state="disabled")

    def load_model(self):
        try:
            self.model = get_saved_or_train_model()
            self.root.after(0, self.ready)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Problem", str(e)))

    def ready(self):
        self.status.config(text="Ready")
        self.btn.config(state="normal")

    def make_row(self):
        try:
            exp = int(self.exp.get())
        except ValueError:
            messagebox.showerror("Input problem", "Experience must be a number.")
            return None

        return pd.DataFrame([{
            "skills": self.skills.get("1.0", "end").strip(),
            "qualification": self.qualification.get().strip(),
            "experience_years": exp,
            "education_level": self.education.get(),
            "language": self.language.get(),
            "certification": self.cert.get(),
            "soft_skills": self.soft.get().strip(),
        }])

    def predict(self):
        row = self.make_row()
        if row is None:
            return

        role, probs = predict_one(self.model, row)

        self.result.config(text=f"Suggested role: {role}")
        self.details.config(state="normal")
        self.details.delete("1.0", "end")

        for _, item in probs.head(5).iterrows():
            percent = round(item["probability"] * 100, 2)
            self.details.insert("end", f"{item['job_role']} - {percent}%\n")

        self.details.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
