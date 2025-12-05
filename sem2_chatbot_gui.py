import tkinter as tk 
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load OFFLINE model from local folder
model = SentenceTransformer('./model')

qa_pairs = [
    ("Hi", "Hello! How can I help you with your Semester 2 preparation today?"),
    ("Hello", "Hi there! Need help with any subject or study tip?"),
    ("Hey", "Hey! Let me know how I can assist with your studies."),
    ("Good morning", "Good morning! Ready to tackle some Semester 2 topics?"),
    ("Good evening", "Good evening! Got any study-related questions for me?"),
    ("How are you?", "I'm just a helpful bot, but I'm always ready to support you!"),
    ("What's up?", "Just here to help you study better! What do you need help with?"),
    ("Who are you?", "I'm Sem2 StudyMate — your friendly exam prep assistant!"),
    ("What can you do?", "I can help you revise concepts, give study tips, and answer common Semester 2 doubts."),
    ("Thank you", "You’re welcome! Always here to help."),
    ("Thanks", "Anytime! Let me know if there’s anything else."),
    ("Bye", "Goodbye! Keep studying smart and take care."),
    ("See you", "See you later! Best of luck with your revision."),
    ("How should I prepare for Engineering Mathematics?", "Start by revising Laplace transforms, differential equations, and vector calculus regularly."),
    ("What are the main topics in MTH102?", "MTH102 includes vector calculus, Laplace transforms, differential equations, and numerical methods."),
    ("I'm stressed about AI/ML. What should I do?", "Break topics into chunks like supervised learning, unsupervised learning, and revise with examples."),
    ("How do I manage time while preparing for multiple subjects?", "Make a weekly schedule. Dedicate fixed slots per subject. Take small breaks."),
    ("Tips for revising Internet and Web Programming?", "Focus on HTML, CSS, JavaScript, and revise PHP basics if included. Practice coding."),
    ("How to prepare for Electronic Systems?", "Understand the basics of diodes, transistors, op-amps, and digital electronics."),
    ("What is the best way to study agriculture subjects?", "Use real-life examples to relate concepts, like seed tech in AGR103 or tools in AGR104."),
    ("How can I reduce stress before exams?", "Sleep well, revise consistently, talk to friends or journal your stress."),
    ("I'm getting confused between all the subjects!", "Try interleaved study: switch subjects every 1–2 hours. Keeps your brain alert."),
    ("Any advice for EVS and Disaster Management?", "Focus on key environmental laws, types of pollution, and basic disaster handling techniques."),
    ("What if I forget everything on exam day?", "That’s common anxiety. Do daily quick reviews to build memory confidence."),
    ("Should I study at night or early morning?", "Choose what works for you. Mornings are better for concentration, nights for creativity."),
    ("How many hours should I study daily?", "Aim for 4–6 focused hours with short breaks. Quality matters more than quantity."),
    ("I can’t focus while studying!", "Use the Pomodoro technique — 25 mins study, 5 mins break. Helps focus improve."),
    ("How to revise long formulas in MTH102?", "Write them down repeatedly, and use mnemonic tricks or sticky notes."),
    ("How to remember definitions in AGR103?", "Use flashcards and quizzes. Teach concepts to a friend."),
    ("How to do well in practical exams?", "Practice experiments or diagrams and understand the theory behind each step."),
    ("Can I listen to music while studying?", "If it helps you focus, yes. Try instrumental or lo-fi beats."),
    ("What if I haven’t started revising yet?", "Don’t panic. Make a 7-day crash plan. Focus on most important topics first."),
    ("Can I study two subjects in one day?", "Yes, in fact it's better. Helps prevent boredom and improves retention."),
    ("How should I start preparing for Semester 2? The syllabus feels huge.", "Start with the subjects you find hardest, and divide your week to cover each one. Make a small 2-week plan first — that’s less overwhelming than a full-semester one."),
    ("I keep mixing up HTML tags and CSS properties. Any trick to remember?", "Try building a small webpage daily using 3–5 tags or properties. Practice by doing — it's the best way to remember web programming."),
    ("What’s the easiest way to revise Engineering Maths quickly?", "Focus on formula sheets and solved examples. Spend more time on Laplace, Fourier, and vector calculus — they carry more weight."),
    ("How do I deal with burnout while studying for 7 subjects?", "Take power breaks — 25 mins study, 5 mins break (Pomodoro method). And mix practical subjects like Web Dev or Electronics with theory ones for variety."),
    ("I'm confused between supervised and unsupervised learning again.", "Think of supervised learning like having a teacher with labeled answers. In unsupervised, the model learns patterns on its own. Clustering is a common example."),
    ("I’m always behind in Environmental Science. What should I focus on?", "Focus on major environmental issues (pollution, global warming, disasters) and case studies. Also revise the disaster management cycle — it’s important."),
    ("Too many formulas in agriculture engineering... how do I remember them?", "Group them topic-wise: irrigation, efficiency, farm machinery. Revise one formula set daily and solve one small numericals set at night."),
    ("Are truth tables really that important for Electronics?", "Yes! Truth tables are the base for logic gate problems. If you get them right, half the question is already solved."),
    ("Sometimes I feel too anxious to even open the book. Any tip?", "Don’t aim for a long study session. Just promise yourself 5 mins with the book. Starting is the hardest part — after that, you’ll get into flow."),
    ("What topics in AI/ML should I not skip?", "Make sure you’re clear on AI types, machine learning basics, KNN, and regression. These are base-level concepts used again in future semesters too."),
    ("I'm fine with theory, but I’m terrible at numericals in maths and agri.", "Start solving 2–3 numericals daily, even if they seem easy. With consistent solving, your confidence builds."),
    ("I forgot most of the Internet protocols. Should I reread or watch videos?", "Try short YouTube videos for revision — especially on TCP/IP, DNS, and HTTP. Then revise your notes right after watching."),
    ("I feel like I’m not making any progress, even though I’m studying.", "Try using a “Done” list instead of a to-do list — write what you finished today. You’re doing better than you think!"),
    ("Do I need to write full answers in EVS or can I use bullet points?", "Use bullet points where possible — clear, concise points with examples often fetch better marks than long paragraphs."),
    ("Is it okay to skip difficult chapters and revise them later?", "Yes — start with what you understand first. Once you build momentum, the tough parts will feel less scary."),
    ("How do I handle similar-sounding soil science terms in AGR103?", "Make a glossary chart with one line definitions. Visual flashcards work great for this kind of subject."),
    ("I'm scared of forgetting everything during the exam.", "That’s common exam anxiety. What helps is writing mini-recalls after each study session — just 3 points you remember."),
    ("Can I skip diagrams in agri engineering?", "Not recommended. Even a rough sketch can earn you marks. Practice them like doodles — speed over perfection."),
    ("What’s the best way to prepare for theory-heavy subjects like EVS and AGR103?", "Make condensed notes in your own words. Revise with peer quizzes — teaching someone else helps you remember better."),
    ("I feel guilty when I take breaks. Is that normal?", "Yes, but breaks are not laziness. They're recovery. A rested brain retains more — especially during back-to-back subject prep."),
    ("How do I manage time when all assignments and exams pile up together?", "Use the 3-2-1 rule: 3 hours for urgent subjects, 2 for revision, and 1 hour for rest or backlog catch-up."),
    ("How many hours a day should I study for Semester 2?", "Quality > quantity. Even 4–5 focused hours with clear goals are better than 10 distracted ones."),
    ("What’s the most important part of the AIML syllabus?", "Understand basic algorithms (KNN, Decision Trees), types of learning, and Python-based implementation ideas."),
    ("How can I improve my Electronics diagrams? They’re messy.", "Use a ruler and draw boxes first, then fill in components. Labeling matters more than neatness."),
    ("My friends are ahead in prep. I feel left behind.", "Everyone has their own pace. Focus on your next step, not someone else’s 10th."),
    ("How can I revise Internet protocols quickly before the exam?", "Make a cheat sheet with key layers (TCP/IP model), ports (like 80 for HTTP), and functions."),
    ("Which chapter in Agri Engineering is most scoring?", "Chapters on tools, machinery classification, and irrigation methods often come with straightforward questions."),
    ("I can’t focus on boring chapters. Any solution?", "Turn it into a challenge: set a timer for 15 mins and try to “beat” it."),
    ("Why does AI have so many confusing definitions?", "That’s normal — AI is interdisciplinary. Focus more on examples than memorizing definitions."),
    ("Can I write point-wise answers in theory subjects?", "Yes! Bullet points, headings, and keywords help the examiner too."),
    ("Should I leave out topics that seem difficult now?", "You can delay, not skip. Bookmark the tough ones, and revisit when your confidence builds."),
    ("How do I revise all subjects in the last week before exams?", "Prioritize! Use a one-page summary per subject and revise MCQs or past papers."),
    ("I panic during viva even if I know the answer.", "Practice saying answers aloud. Record yourself or ask a friend to quiz you casually."),
    ("I forget machine names and functions in agri subjects. What to do?", "Make flashcards with images. Even hand-drawn ones work."),
    ("Is it okay if I can’t complete 100% syllabus?", "Yes. Focus on high-weightage topics and revise them well."),
]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
questions = [q for q, _ in qa_pairs]
answers = [a for _, a in qa_pairs]
question_embeddings = model.encode(questions)
def get_response(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    if similarities[0][best_match_idx] < 0.5:
        return "Hmm... I'm not sure about that. Could you rephrase?"
    return answers[best_match_idx]
def send_message():
    user_input = entry.get().strip()
    if user_input == "":
        return
    chat_window.config(state='normal')
    chat_window.insert(tk.END, "You: " + user_input + "\n")
    response = get_response(user_input)
    chat_window.insert(tk.END, "Sem2 StudyMate: " + response + "\n\n")
    chat_window.config(state='disabled')
    entry.delete(0, tk.END)
    chat_window.see(tk.END)
root = tk.Tk()
root.title("Sem2 StudyMate")
root.geometry("500x600")
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 11))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
entry_frame = tk.Frame(root)
entry_frame.pack(fill=tk.X, padx=10, pady=10)
entry = tk.Entry(entry_frame, font=("Arial", 12))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry.bind("<Return>", lambda event: send_message())
send_button = tk.Button(entry_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT)
root.mainloop()