from tkinter import *
from mydb import Database
from tkinter import messagebox
from api import API
class NlpApp:
    def __init__(self):
        self.obj=Database()
        self.api=API()
        self.root=Tk()
        self.root.title("NLP App")
        self.root.iconbitmap("resources/favicon.ico")
        self.root.geometry("350x600")
        self.root.configure(background="#282224")
        self.login_gui()
        self.root.mainloop()
    def login_gui(self):
        self.clear()
        heading=Label(self.root,text="NLP App",bg="#282224",fg="white")
        heading.pack(pady=(30,30))
        heading.configure(font="verdana 30 bold")
        label1=Label(self.root,text="Enter Email",bg="#282224",fg="white")
        label1.pack(pady=(30,15))
        label1.configure(font="verdana 10 bold")
        self.email_entry=Entry(self.root,width=40)
        self.email_entry.pack(pady=(2,4),ipady=5)
        label2= Label(self.root, text="Enter Password", bg="#282224", fg="white")
        label2.pack(pady=(30, 15))
        label2.configure(font="verdana 10 bold")
        self.password_entry = Entry(self.root, width=40,show="*")
        self.password_entry.pack(pady=(2, 4), ipady=5)
        login_button= Button(self.root,text="Login",width=20,height=2,command=self.login)
        login_button.pack(pady=(40,10))
        label3 = Label(self.root, text="Not a member?", bg="#282224", fg="white")
        label3.pack(padx=50,pady=(40,5))
        label3.configure(font="verdana 10 bold")
        register_button=Button(self.root,text="Register Now",command=self.register_gui)
        register_button.pack(pady=(10,10))

    def register_gui(self):
        self.clear()
        heading=Label(self.root,text="NLP App",bg="#282224",fg="white")
        heading.pack(pady=(30,30))
        heading.configure(font="verdana 30 bold")
        label0= Label(self.root, text="Enter Name", bg="#282224", fg="white")
        label0.pack(pady=(30, 15))
        label0.configure(font="verdana 10 bold")
        self.name=Entry(self.root,width=40)
        self.name.pack(pady=(2,4),ipady=5)
        label1=Label(self.root,text="Enter Email",bg="#282224",fg="white")
        label1.pack(pady=(30,15))
        label1.configure(font="verdana 10 bold")
        self.email_entry1=Entry(self.root,width=40)
        self.email_entry1.pack(pady=(2,4),ipady=5)
        label2= Label(self.root, text="Enter Password", bg="#282224", fg="white")
        label2.pack(pady=(30, 15))
        label2.configure(font="verdana 10 bold")
        self.password_entry1 = Entry(self.root, width=40,show="*")
        self.password_entry1.pack(pady=(2, 4), ipady=5)
        register_button= Button(self.root,text="Register",width=20,height=2,command=self.register)
        register_button.pack(pady=(40,10))
        label3 = Label(self.root, text="Already have an account?", bg="#282224", fg="white")
        label3.pack(padx=50,pady=(20,5))
        label3.configure(font="verdana 10 bold")
        register_button=Button(self.root,text="Login",command=self.login_gui)
        register_button.pack(pady=(10,10))

    def clear(self):
         for i in self.root.pack_slaves():
             i.destroy()
    def register(self):
        name=self.name.get()
        email=self.email_entry1.get()
        password=self.password_entry1.get()
        response=self.obj.add_value(name,email,password)
        if response:
            messagebox.showinfo("Success","now you can login")
        else:
            messagebox.showerror("Error","email already exists")
    def login(self):
        email=self.email_entry.get()
        password=self.password_entry.get()
        response=self.obj.login(email,password)
        if response:
            messagebox.showinfo("Success","login successful")
            self.options()
        else:
            messagebox.showerror("Error","login failed")
    def options(self):
        self.clear()
        heading = Label(self.root, text="NLP App", bg="#282224", fg="white")
        heading.pack(pady=(30, 30))
        heading.configure(font="verdana 30 bold")
        self.ner=Button(self.root,text="Summarization",width=30,height=5,command=self.ner)
        self.ner.pack(pady=(10,10))
        self.sentiment_analysis = Button(self.root, text="Sentiment Analysis", width=30, height=5,command=self.sentiment_analysis2)
        self.sentiment_analysis.pack(pady=(15, 10))
        self.headline1= Button(self.root, text="Headline Generation", width=30, height=5,command=self.headline)
        self.headline1.pack(pady=(15, 10))
        self.logout=Button(self.root,text="Logout",width=10,command=self.login_gui)
        self.logout.pack(pady=(20,10))

    def ner(self):
        self.clear()
        heading = Label(self.root, text="NLP App", bg="#282224", fg="white")
        heading.pack(pady=(30, 30))
        heading.configure(font="verdana 30 bold")
        heading2 = Label(self.root, text="Summarization", bg="#282224", fg="white")
        heading2.pack(pady=(30, 30))
        heading2.configure(font="verdana 30")
        label=Label(self.root,text="Enter the text",bg="#282224",fg='white')
        label.pack(pady=(10,10))
        self.text=Entry(self.root,width=40)
        self.text.pack(pady=(10,10),ipady=30)
        self.label3 = Label(self.root, text="", bg="#282224", fg='white')
        self.label3.pack(pady=(10, 10))
        self.ner_button=Button(self.root,text="Analyze",width=20,height=2,command=self.summarization)
        self.ner_button.pack(pady=(10,10))
        self.go_back = Button(self.root, text="Go Back", width=10, command=self.options)
        self.go_back.pack(pady=(20, 10))
    def summarization(self):
        text=self.text.get()
        response=self.api.ner(text)
        try:
           self.label3["text"]=response
        except:
            messagebox.showerror("Error","Summarization failed")
    def sentiment_analysis2(self):
        self.clear()
        heading = Label(self.root, text="NLP App", bg="#282224", fg="white")
        heading.pack(pady=(30, 30))
        heading.configure(font="verdana 30 bold")
        heading2 = Label(self.root, text="Sentiment Analysis", bg="#282224", fg="white")
        heading2.pack(pady=(30, 30))
        heading2.configure(font="verdana 30")
        label = Label(self.root, text="Enter the text", bg="#282224", fg='white')
        label.pack(pady=(10, 10))
        self.text4 = Entry(self.root, width=40)
        self.text4.pack(pady=(10, 10), ipady=30)
        self.label4 = Label(self.root, text="", bg="#282224", fg='white')
        self.label4.pack(pady=(10, 10))
        self.sentiment_analysis = Button(self.root, text="Analyze", width=20, height=2, command=self.sentiment)
        self.sentiment_analysis.pack(pady=(10, 10))

        self.go_back = Button(self.root, text="Go Back", width=10, command=self.options)
        self.go_back.pack(pady=(20, 10))
    def sentiment(self):
        text=self.text4.get()
        response=self.api.sentiment(text)
        try:
            text=""
            for i in response["scored_labels"]:
                text=text+i["label"]+"->"+str(i["score"])+" "
            self.label4["text"]=text
        except:
            messagebox.showerror("Error","Sentiment failed")

    def headline(self):
        self.clear()
        heading = Label(self.root, text="NLP App", bg="#282224", fg="white")
        heading.pack(pady=(30, 30))
        heading.configure(font="verdana 30 bold")
        heading2 = Label(self.root, text="Headline Generation", bg="#282224", fg="white")
        heading2.pack(pady=(30, 30))
        heading2.configure(font="verdana 30")
        label = Label(self.root, text="Enter the text", bg="#282224", fg='white')
        label.pack(pady=(10, 10))
        self.text5 = Entry(self.root, width=40)
        self.text5.pack(pady=(10, 10), ipady=30)
        self.label5 = Label(self.root, text="", bg="#282224", fg='white')
        self.label5.pack(pady=(10, 10))
        self.headline_generation2 = Button(self.root, text="Analyze", width=20, height=2, command=self.headline_generation)
        self.headline_generation2.pack(pady=(10, 10))

        self.go_back = Button(self.root, text="Go Back", width=10, command=self.options)
        self.go_back.pack(pady=(20, 10))

    def headline_generation(self):
        text=self.text5.get()
        response=self.api.headline(text)
        try:
            self.label5["text"]=response["summary_text"]
        except:
            messagebox.showerror("Error","headline generation failed")
NlpApp()
