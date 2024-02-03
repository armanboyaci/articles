# Resources 

## DESIGN
[No Silver Bullet, Fred Brook (1986)](http://worrydream.com/refs/Brooks-NoSilverBullet.pdf)

The hardest single part of the building software system is deciding precisely what to build. The clients do not know what they wants. 
Therefore the most important function that software builders do for their clients is the iterative extraction and refinement of the product requirements.

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

## TEAM ORGANIZATION

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

## UNDERSTANDING

[Up and Down Ladder of Abstraction, Bret Victor (2011)](http://worrydream.com/#!2/LadderOfAbstraction)

How can we design systems when we don't know what we're doing? The most exciting engineering challenges lie on the boundary of theory and the unknown. When designing at this boundary, the challenge lies not in constructing the system, but in understanding it. In the absence of theory, we must develop an intuition to guide our decisions. The design process is thus one of exploration and discovery. The most powerful way to gain insight into a system is by moving between levels of abstraction. Many designers do this instinctively. But it's easy to get stuck on the ground, experiencing concrete systems with no higher-level view. It's also easy to get stuck in the clouds, working entirely with abstract equations or aggregate statistics. This interactive essay presents the ladder of abstraction, a technique for thinking explicitly about these levels, so a designer can move among them consciously and confidently.


[Visualization, modeling, and surprises, John D. Cook (2013)](https://www.johndcook.com/blog/2013/02/07/visualization-modeling-and-surprises/)

Visualization can show you something in your data that you didn’t expect. But some things are hard to see, and visualization is a slow, human process. Modeling might tell you something slightly unexpected, but your choice of model restricts what you’re going to find once you’ve fit it. So you iterate. Visualization suggests a model, and then you use your model to factor out some feature of the data. Then you visualize again.
