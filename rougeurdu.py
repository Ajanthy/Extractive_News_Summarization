from rouge import Rouge

rouge = Rouge()


# system_summary ="But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake."
# reference_summary = "TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn.For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn.Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier.However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues.Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters.Time Warner's fourth quarter profits were slightly better than analysts' expectations."
#
# text = """However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people.
# It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream.
# Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”
# I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger.
# “Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
# Develop A Powerful Vision Of What You Want
# “Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
# I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator.
# Commit to it."""
#
# text2 = """ Have you experienced this before? Who is right and who is wrong? Neither. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It must come from within you. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? So become intentional on what you want out of life. Commit to it. Nurture your dreams.
# """

# reffile = open('politicalsummary.txt', "rt")
# # Title = reffile.readline().rstrip()  # Discard first line
# reference = reffile.read()
#
# # hypfile = open('sampleFeatures.txt', "rt")
# # # Title = txtfile.readline().rstrip()  # Discard first line
# # hypotheses = hypfile.read()
#
# hypfile = open('finaloutput.txt', "rt")
# # Title = txtfile.readline().rstrip()  # Discard first line
# hypotheses = hypfile.read()

def evaluate(hypfile, reffile):
    texthypFile = open(hypfile, "rt")
    textreffile = open(reffile, "rt")
    reference = textreffile.read()
    hypotheses = texthypFile.read()
    scores = rouge.get_scores(hypotheses, reference)
    print("Score for ", hypfile, " : ", scores)
    return scores
