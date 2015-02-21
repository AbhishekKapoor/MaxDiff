design <- function (attributes,sets,samplesize){
  
  if (sets > attributes/2) 
    stop ("We recommend displaying either 4 or 5 sets to each respondents.However, we don't recommend
          sets greater than half of the attributes")
  samplesize<- version
  lowerlimit<- 2* attributes/sets
  ideal<- 3* attributes/sets
  upperlimit<- 4* attributes/sets
  des<- tradeoff.des (attributes,sets)
  
}