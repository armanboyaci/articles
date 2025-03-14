# Resources 

## DESIGN
[No Silver Bullet, Fred Brook (1986)](http://worrydream.com/refs/Brooks-NoSilverBullet.pdf)

The hardest single part of the building software system is deciding precisely what to build. The clients do not know what they wants. 
Therefore the most important function that software builders do for their clients is the iterative extraction and refinement of the product requirements.

Great design comes from great designers. Software construction is a creative process. [...] very best designers produce structures that are faster, smaller, simpler, cleaner, and produced with less effort. The differences between the great and average approach an order of magnitude.

[Hammock Driven Development, Hickey Rich (2010)](https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/HammockDrivenDev-mostly-text.md)

We should be solving problems.
Analysis and Design is about two things: identifying some problem that we are trying to solve, and assessing our proposed solution in terms of whether or not it solves that problem. 

The least expensive place to fix bugs is when you are designing your software. 
Problems of misconception are not generally addressed by testing, or type systems, or the things we use to correct defects in implementation. 
There are not really type systems that can tell us if we have got a good idea, or what we are doing addresses that idea. 

[Simple Made Easy, Hickey Rich (2011)](https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/SimpleMadeEasy.md)

We need to build simple systems if we want to build good systems. Benefits of simplicity: easy to understand, easy to change, easy to debug.

[The Most Important Design Guideline, Scott Meyers (2013)](https://www.youtube.com/watch?v=sfLZ7v9gEnc)

Make interfaces easy to use correctly and hard to use incorrectly: Choose good names, be consistent, progressive disclosure, documentation.

[Edge Case Poisoning, Hillel Wayne (2020)](https://buttondown.email/hillelwayne/archive/edge-case-poisoning/)

Designing a system to handle the edge cases of a client may punish other clients. Simply because now the system is over complicated for others.

[The Artist and The Innovator, Ash Maurya (2018)](https://blog.leanstack.com/the-artist-and-the-innovator/)

Customers don’t care about your solution but their problems. The challenge today isn’t building a product but uncovering the right product to build.

[Dimensions Of Programming, Peter Norvig (2012)](https://www.youtube.com/watch?v=nEfMvR2OMcM)

In real world, software programs evolve over time. We can *plot* the evolution of a program through a multidimensional space: (1) correctness, (2) efficiency, (3) features, (4) elegance (clarity, simplicity and generality). We select a dimension and improve our program on that direction. 

Changing the elegance will not improve correctness/efficientcy/features. It is for gaining for the future, a program more elegant is easier to maintain/change.

> The best is the enemy of the good. --Voltaire

Do not spend too much time for perfection. Sometimes it would be much valuable to spend time to improve efficiency of the program instead of chasing 100% correctness. A good engineer makes good trade-offs. Which direction do I need to move, or am I done?

[How to Build Good Software (2019)](https://knowledge.csc.gov.sg/ethos-issue-21/how-to-build-good-software/)

The main value in software is not the code produced, but the knowledge accumulated by the people who produced it. 

IT systems are often full of features but are still hated by users because of how confusing they become. In contrast, highly ranked mobile apps tend to be lauded for their simplicity and intuitiveness. Building good software involves alternating cycles of expanding and reducing complexity. 

As new features are developed, disorder naturally accumulates in the system. When this messiness starts to cause problems, progress is suspended to spend time cleaning up. Software should be treated not as a static product, but as a living manifestation of the development team’s collective understanding. 

Software projects rarely fail because they are too small; they fail because they get too big. Unfortunately, keeping a project focused is very hard in practice: just gathering the requirements from all stakeholders already creates a huge list of features.

Small teams of the best engineers can often build things faster than even very large teams of average engineers. In large projects, bad engineers end up creating more work for one another, as errors and poor design choices snowball to create massive issues

[The Value of Source Code (2024)](https://www.youtube.com/watch?v=Y6ZHV0RH0fQ)

Writing new code is usually faster than understanding and modifying existing code. Production view vs theory building view. In production view the output is the code, in theory building view the real output is the map between the real-world and the code. The code and its documentation has proved insufficient as a carrier of some of the most important design ideas. So much value is tied up in individual contributor. Companies shouldn't treat software developers as replaceable cogs in a machine their domain knowledge acquired over the years cannot be easily repleaced.

## DATA SCIENCE

[Tukey Design Thinking and Better Questions, Roger Peng (2019)](https://web.archive.org/web/20190418141631/https://simplystatistics.org/2019/04/17/tukey-design-thinking-and-better-questions/)

(In data science) We almost always start with a vague and poorly defined question and a similarly vague sense of what procedure to use. 
The most useful thing a data scientist can do is to devote serious effort towards improving the quality and sharpness of the question being asked.

[When Better Isn't Better?, Jean-François Puget (2012)](https://web.archive.org/web/20151001075209/https://www.ibm.com/developerworks/community/blogs/jfp/entry/when_better_isn_t_better?lang=en)

Customers value solutions that are locally optimal. 
Before showing a solution to the customer, try easy improvement schemes. 
It will ensure that the optimization process outputs solutions that aren't easily improved.  

[Setting Expectations in Data Science Projects (2012)](https://win-vector.com/2012/04/21/setting-expectations-in-data-science-projects/)

“We better at least quantify expected performance: let’s insist on an accuracy of 95%.” 
Unfortunately this “bar” is often set without any research if accuracy is even the correct measure, 
if 95% is easy or hard or even if the enterprise will be profitable at this accuracy. 
What we need to do is: schedule dedicated time to learn about the domain and data before writing project goals and scope. 
This itself can be part of a small concrete expectation setting project. 
To complete the expectation setting project we need reusable methods to set useful, realistic goals that really measure if a data science project is on track. 
An initial research project to quantify what sort of outcome is even possible and if such an outcome would make sense for the business. 

[Change Management for Advanced Analytics Projects (2017)](https://www.princetonoptimization.com/blog/blog/change-management-advanced-analytics-projects)

We can classify analytics projects into one of three buckets:
1. A one-time analysis leading to a strategic decision
2. Analytics that are used to drive repeated decisions with humans in the loop
3. Analytics that are deeply embedded in automated process.

Collect answers to questions such as:
1. The problem statement and usability requirements: What is the current situation? What is your vision of the ideal situation? What actions would help to bridge the gap? How will success be measured? Is there a baseline measure available today that can be used for comparison? What is the cost of doing nothing?
2. Identify the stakeholders: Who are the users of a potential solution? Who inside and outside the organization will be impacted by the potential solution? Which people from different parts of the organization need to be involved in the construction of a potential solution? These could be end users, IT executives, database architects, and user interface designers. What business processes will be affected by the potential solution, and how?
3. Obtain stakeholder agreement on the problem statement. Without agreement at the beginning from all of the various stakeholders as to the problem being addressed, there will be challenges in moving to the acceptance and commitment phases of the change management process.

It is better to tell users, “The system will help you make decisions faster” instead of “The system will make better decisions for you.”

For systems where users will interact with the recommendations made by analytics, create controls that allow users to override those recommendations. Over time, users will trust the system’s recommendations. Make sure to build the system to log these overrides, as that data can indicate how to improve the system.

Create a Graphical User Interface (GUI) of the current process, with connections to data and illustrations of current decisions. 
This will allow users to get comfortable with a new system.

To move towards the commitment phase, incremental deployment of a solution is a best practice. Increase the scope across appropriate business dimensions: start with a single product, single business unit, and/or a single geography; then expand the scope incrementally.

It is critical to continually measure the benefits of the analytics. During solution discovery, define the metrics that can be measured that demonstrate success. During deployment, report on these measures.


[Be prepared to show your working!, David Spiegelhalter (2019)](https://www.youtube.com/watch?v=E12_F4xeOHw)

The users of an AI system should expect trustworty claims: (1) by the system, (2) about the system by the company/devs. Variaty of audiences and purposes for explainability/interpretability.

A structure for evaluation:
- Phase 1. Digital testing: performance
- Phase 2. Lab testing: user testing
- Phase 3. Field testing: controlled trials of impact (randomized roll-out)
- Phase 4. Routine use: monitoring

Intelligent Openness of information/claims: accessible, intelligable (easy to understand), useable, assessable.
Principles for Accountable Algorithms: Accuracy, Explainability, Impact

(Local) Explailability:
- What drove the conclusion? What was the chain of reasoning
- What-if inputs had been different? Counterfactuals
- How confident is the conclusion?
- Is the current situation withi its competence?


## CODE ORGANIZATION

[The Clean Architecture, Uncle Bob (2012)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
1. Independent of Frameworks. The architecture does not depend on the existence of some library of feature laden software. This allows you to use such frameworks as tools, rather than having to cram your system into their limited constraints.
2. Testable. The business rules can be tested without the UI, Database, Web Server, or any other external element.
3. Independent of UI. The UI can change easily, without changing the rest of the system. A Web UI could be replaced with a console UI, for example, without changing the business rules.
4. Independent of Database. You can swap out Oracle or SQL Server, for Mongo, BigTable, CouchDB, or something else. Your business rules are not bound to the database.
5. Independent of any external agency. In fact your business rules simply don’t know anything at all about the outside world. 

[Indirection is not abstraction, Silas Reinegal (2018)](https://www.silasreinagel.com/blog/2018/10/30/indirection-is-not-abstraction/)

Indirectness is about easy to change. Abstraction is about generalization.

Low coupling exists where things are indirectly connected. 
Things that are directly connected are simple, faster to build, and easier to understand. 
However, they are inflexible. When there are expected dimensions of software change, it can be worth paying the cost to create software with higher flexibility. 
Flexible software is harder to create, more complex, harder to understand, and harder to maintain.

However, for details at the same level of abstraction, there is no inherent virtue in adding layers of indirection. 
Details are absolutely allowed to depend on other details at the same level of abstraction. 
There is no merit in adding in interfaces and separating them out into separate files, unless that specific flexibility is needed.

The Dependency Inversion Principle is about abstraction, not about indirection (Indirection doesn’t increase the usability (or re-usability) of software.):   
High-level modules should not depend on low-level modules. Both should depend on abstractions.

[Indirecation is not abstraction, Zed A. Shaw (2014)](https://web.archive.org/web/20160304022133/https://zedshaw.com/archive/indirection-is-not-abstraction/)

Abstraction is used to reduce complexity. Indirection is used to reduce coupling or dependence. 
Abstraction: the process of separating the interface to some functionality from the underlying implementation in such a way that the implementation can be changed without changing the way that piece of code is used.
Abstraction’s simplicity goal competes with how indirection complicates things.

[Build Abstractions Not Illusions, Gregor Hohpe (2023)](https://www.youtube.com/watch?v=aWZFRk-w3ng)

We sometimes confuse abstraction and composition. Naming something by the pieces that it is made out of is not a useful abstraction. A useful abstraction should bring a new vocabulary that shields the user from the underlying complexity. Composition is useful as well but it needs you to familiar with the lower layer vocabulary. When we hide the essential details we create a dangerous *illusion*. What is essential (and what is not) is a judgement call, it depends on the context, somebody (the architect?) must decide what should we hide and expose. 

Just providing a system (for example a database) with sane defaults is not useful to the user. The platform team should bring a new language (closer the the business language) for the user.

[Bottleneck #01: Tech Debt, Tim Cochran and Carl Nygard (2022)](https://martinfowler.com/articles/bottlenecks-of-scaleups/01-tech-debt.html)

We use the term *tech debt* for different things: 
1. Code quality
2. Testing
3. Coupling
4. Unused or low value features
5. Out of date libraries or frameworks
6. Tooling
7. Reliability and performance engineering problems
8. Manual processes
9. Automated deployments
10. Knowledge sharing


## TEAM ORGANIZATION

[Steve Jobs on Consulting, Steve Jobs](https://www.youtube.com/watch?v=-c4CNB80SRc)
Making recommendations without owning the implementation and results captures only a fraction of the value of the opportunity to learn the business. Without the experience of actually doing it, you never truly understand it.

[Managers, Marketing, and Continuous Process Improvement, Steve Jobs](https://www.youtube.com/watch?v=rQKis2Cfpeo)
The great people are self-managing, they don't need to be managed. Once they know what to do, they will figure out how to do it. What they need is a common vision. And that is leadership. If you are great person, why do you want to work for somebody that you can't learn anything from. 

[Essense of Product Management, Steve Jobs](https://www.youtube.com/watch?v=XmRNIGqzuRI)
Designing a product is to choose what to emphesize and what to remove it. Different companies are making different choices and market decides who is right at the end. The customers are paying us to make those choices.

[Managing Peoplei Steve Jobs](https://www.youtube.com/watch?v=f60dheI4ARg)
You need to be great at divide things up, to have great teams, touch bases frequently.

[Great idea doesn't always translates into great product, Steve Jobs](https://www.youtube.com/watch?v=Qdplq4cj76I)
Thinking that having an idea is 90% of the work is a mistake. There is a great deal of craftsmanship between a great idea and a great product. The final product never fully resembles the initial idea because you learn so much throughout the process of making it. A group of talented people refines and builds upon each other's ideas, resulting in a beautiful product.

[Team Interaction Modeling with Team Topologies (2023)](https://teamtopologies.com/key-concepts-content/team-interaction-modeling-with-team-topologies)

The team is the smallest entity of delivery within an organization and should be a stable group of 5 to 9 people working towards a shared goal as a unit. 

Reducing the cognitive load of a team through: 1) a Platform service, 2) a complicated subsystem service, 3) an Enabling team.

Hand-overs between teams slow teams down and create increased work-in-progress as each team picks up something else whilst waiting on the next team to complete their work. 
It is important that we monitor the interactions between teams and try to guide the inter-team collaboration to ensure that it does not occur over prolonged periods unless that is desired. 
Although collaboration is good for discovery, it can also be expensive. 
It is, therefore, important that any frequent or prolonged periods of collaboration are performed 
to explore (and possibly create) an X-as-a-Service interaction model between the two teams for future interactions. 
Stream-aligned teams should generally never provide an X-as-a-Service directly. Instead, data or services from the Stream-aligned team should be made available ‘as a Service’ via a platform.


[Broken Ownership, Alex Everlof (2023)](https://blog.alexewerlof.com/p/broken-ownership)

For true ownership: mandate + responsibility + knowledge

1. You cannot be responsible for something you don’t control. You need the mandate.
2. You cannot use that mandate effectively over something you don’t understand. You need knowledge.
3. You gain experience from running something and learn from the mistakes when things go wrong. You don’t use that knowledge to optimize it if it’s not your problem. You need to be held responsible.

[The death of Agile, Allen Holub (2015)](https://www.youtube.com/watch?v=vSnCeJEka_s)

You don't get a good result by following a specific process.
We need flexibiltiy and abilitiy to change and many corporations are incapable of doing. Agility is about eliminating friction. 
The entire organization should be agile, not teams. If you change the process, you also need to change the organization.
For agility to work you need to put proper culture in place, it has to come from top-to-down. It is CEO's job to manage the culture of the organization so if the CEO does not understand what the agility is about.
Trust is everything is agile, trusting people to do their job.
The product is developed with the customers who are talking with you day-to-day basis and product owners are not customers. (Instead of negiotate) collaborate with customers.

[Scaling agile has never worked ... and never will, Geoff Watts (2021)](https://www.youtube.com/watch?v=YhUY5olYNB4)

The mechanical metaphor that organizations have traditionally been built on focused  on taking inputs and efficiently turning them into nice packaged repeatable predictable outputs through lots of structure is not the right metaphor if we require effectiveness instead of efficiency. The organic metaphor for agile where we focus on principles, values, rituals and collaboration.
Agile is a solution to a problem but not the only solution to all problems. If you kept on focusing on effectivenss over efficiency we will never take advantage of our innovation, get profit. So it is a balance.

If we are not getting right results we need to change people's actions. But the actions are based on our beliefs and our beliefs are formed and reinforced by our experiences. We miss this part because they are not as visible as results and actions. The environment is complex and unpredictable so the focus of leadership should not be standardizing and correcting actions but mentoring and coaching people to create experiencers that will lead to coherent beliefs so that actions can be autonomous and coherent.

[What we don't understand about trust](https://www.ted.com/talks/onora_o_neill_what_we_don_t_understand_about_trust)

Trust is the response to trustworthiness. We can control our trustworthiness, but trust is given by others, so it is not in our control. We need to communicate our trustworthiness to other people. We have to provide evidence that we are trustworthy. We trust someone to do something. For example, we trust a professor in teaching mathematics but maybe not in driving the school bus. Trustworthiness has three dimensions: (1) competence, (2) honesty, and (3) reliability.

## UNDERSTANDING

[Up and Down Ladder of Abstraction, Bret Victor (2011)](http://worrydream.com/#!2/LadderOfAbstraction)

How can we design systems when we don't know what we're doing? The most exciting engineering challenges lie on the boundary of theory and the unknown. When designing at this boundary, the challenge lies not in constructing the system, but in understanding it. In the absence of theory, we must develop an intuition to guide our decisions. The design process is thus one of exploration and discovery. The most powerful way to gain insight into a system is by moving between levels of abstraction. Many designers do this instinctively. But it's easy to get stuck on the ground, experiencing concrete systems with no higher-level view. It's also easy to get stuck in the clouds, working entirely with abstract equations or aggregate statistics. This interactive essay presents the ladder of abstraction, a technique for thinking explicitly about these levels, so a designer can move among them consciously and confidently.


[Visualization, modeling, and surprises, John D. Cook (2013)](https://www.johndcook.com/blog/2013/02/07/visualization-modeling-and-surprises/)

Visualization can show you something in your data that you didn’t expect. But some things are hard to see, and visualization is a slow, human process. Modeling might tell you something slightly unexpected, but your choice of model restricts what you’re going to find once you’ve fit it. So you iterate. Visualization suggests a model, and then you use your model to factor out some feature of the data. Then you visualize again.
