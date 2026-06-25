library(irace)
scenario <- readScenario(filename = "scenario.txt")
scenario$recoveryFile <- "irace_ga_recovered.Rdata"
scenario$logFile <- "irace_ga_recovered_v2.Rdata"
scenario$parallel <- 6
irace(scenario = scenario)