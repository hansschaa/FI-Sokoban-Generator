library(irace)
scenario <- readScenario(filename = "scenario.txt")
scenario$recoveryFile <- "irace.Rdata"
scenario$logFile <- "irace_recovered.Rdata"
irace(scenario = scenario)
