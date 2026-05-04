#import "@preview/elsearticle:1.0.0": *
#import "@preview/mannot:0.3.1": *
#import "@preview/wordometer:0.1.5": word-count, word-count-of, total-words, total-characters

//#set page(numbering: "1 of 1")
//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

#show: elsearticle.with(
  title: "Topology optimization hand-in",
  authors: (
    (
      name: "Sigurd Mousten Jager Nielsen",
      affiliation: "Aarhus Universitet",
      corr: "202108107@uni.au.dk",
      id: none,
    ),
  ),
  journal: none,
  abstract: none,
  keywords: none,
  format: "preprint",
  // line-numbering: true,
)

//AU-LOGO
#figure(
  image("images/au_logo.png", width: 25%),
)
/*
#show: word-count.with(exclude: ())
#align(center)[In this hand-in there are #total-words words and #total-characters characters in sections @section_1 through @section_5]
*/
//PAGEBREAK
#colbreak()

