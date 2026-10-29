# Video Presentation Script (5 speakers, about 8 minutes)

Each slide's speaker notes in the PowerPoint contain the same text. Recording tip: read slowly, one speaker per slide section.


## Speaker 1 – Calug, Adrian Samuel (about 1:30)

**Slide 1** – *Cue: Show title slide; soft background music starts (low volume).*

Good day everyone! We are the BS Mathematics 3rd Year, Section B group, and today we will show how a computer can help teachers notice struggling students early, before it is too late to help. Our topic is: Predicting Student At-Risk Status with a Decision Tree.

**Slide 2** – *Cue: B-roll idea: classroom or students studying (add your own photo/video).*

Imagine a teacher with two hundred students. By the time the final grades come out, it is already too late to help anyone. In the dataset we studied, one million students, about five out of every hundred ended with a D or F. What if we could raise a flag early? That is exactly what decision trees are good at. You already meet them in real life: schools use early-warning flags, hospitals ask step-by-step triage questions, banks follow steps to approve loans, and support desks route your problem by asking a few questions. Our project uses one to predict who may fail, using only three habits: weekly study hours, class attendance, and class participation.


## Speaker 2 – Hilario, Angelo (about 1:30)

**Slide 3** – *Cue: Point to each box of the tree while speaking.*

So what is a decision tree? Think of a game of twenty questions. The computer asks a yes-or-no question, follows the answer, asks another, and finally reaches a decision. Here is a simplified example from our model. First question: does the student study 4 hours a week or less? If yes, we raise a red flag. If no, we ask a second question: 6 hours or less? If yes, we keep an eye on the student. Otherwise the student looks fine. How does the tree choose its questions? It picks the question that makes the groups less mixed. We measure how mixed a group is with Gini impurity. On the right, zero means everybody in the group has the same outcome, and the highest point means a perfect fifty-fifty mix. The big advantage is that a teacher can read every rule. Our real tree has five levels and thirty-two endings, but it works the same way.


## Speaker 3 – Jaculina, Adrienne Christopher (about 1:30)

**Slide 4** – *Cue: Point to the three colored blocks of the split bar.*

Now, how did we build and test it? We started with one million student records. Important detail: we did not use the total score, because the grade is computed from it. That would be like peeking at the answer key. Next, we split the students into three groups, like a school year: seventy percent for training, which is studying; fifteen percent for validation, which is the practice test; and fifteen percent for the final test, which we open only once. We tuned the tree with five-fold cross-validation and chose the simplest good one, five levels deep. Then, using only the practice group, we chose two alert levels: a balanced alert, and a sensitive alert that prefers catching more students even if there are more false alarms, because missing a struggling student costs more than checking on a student who is fine.


## Speaker 4 – Mendoza, Clarence Dion Reigh (about 2:00)

**Slide 5** – *Cue: Point to the TP/FN/FP/TN counts.*

Now the results, on 150,000 students the model had never seen. With the balanced alert, the tree catches 61 out of every 100 at-risk students, and about half of the students it flags really are at risk. With the sensitive alert, it catches 78 out of 100, with more false alarms. The matrices on the left show these counts. We also counted the cost of mistakes, treating a missed student as five times worse than a false alarm. Compared with flagging nobody, the sensitive alert cuts the mistake cost by 54 percent. So the tool is useful as a way to start a conversation with a student, not as a final verdict.

**Slide 6** – *Cue: Point to the red dots, then the bars.*

What drives the prediction? The scatter plot shows that students who study more tend to score higher, and the red dots below the line are the at-risk students. The bar chart shows study hours account for almost all of the model's decisions. Attendance and participation are close to zero. That surprised us, and it is suspicious: real classrooms usually show a link with attendance. Together with other odd patterns, such as a quarter of all scores being exactly one hundred, it suggests this dataset is simulated. So please do not conclude that attendance does not matter in real life; this is a limit of our data.


## Speaker 5 – Yumul, Aries (about 1:15)

**Slide 7** – *Cue: Show GitHub repository on screen; music fades out.*

To conclude: we built a simple, readable decision tree that flags students who may end with a D or F. It matched a more complex baseline, logistic regression, while staying fully explainable. Study hours was the only useful clue in this dataset. We also learned that choosing the alert level should depend on the cost of mistakes, not only on accuracy. Used in real schools, this tool must be fair, private, and always reviewed by a teacher, and it should lead to support, never punishment. Next steps are to test on real school data with prior grades, check fairness across student groups, and ask teachers what false-alarm cost is acceptable. Thank you for listening. Our code and paper are in our GitHub repository. We welcome your questions.


## Visual aids and music to add (not included in the file)

- Slide 2: a classroom photo or short clip (e.g., royalty-free from Pexels or Pixabay).
- Slide 3: optional short screen recording of the tree rules in `results/tree_rules.txt`.
- Background music: a calm royalty-free instrumental track at low volume; credit the source in the video's end screen.
- Charts and diagrams are already on the slides (from the project's `figures/` folder).
