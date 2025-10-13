/*******************************************************************************
* ECON 5361 Final Project do-file (Group 8)
* Author information redacted
* Created: 04/05/2025
*******************************************************************************/

* Clear workspace and disable paging
clear all
set more off

* Set working directory (update to your local project path)
* cd "/path/to/your/project"

* Close any open log file and open a new one
cap log close           
log using "final_log.log", replace

* Set figure style

set scheme sj // Set Stata Journal style
grstyle init // Initialization
grstyle set plain, horizontal grid box // Set background style

* Load the data
use "final_data.dta", clear

********************************************************************************
* DATA CLEANING AND VARIABLE CREATION
********************************************************************************

* Summary statistics of raw data
summarize

* Recode values that indicate "Not in Universe" or missing values to actual missing
replace inctot = . if inctot >= 99999990  // NIU/missing codes
replace incwage = . if incwage >= 99999990
replace age = . if age == 99  // 99 is 99+ years in some years, NIU/missing in others
replace citizen = . if citizen == 9  // 9 is NIU
replace sex = . if sex == 9  // 9 is NIU according to docs
replace race = . if race == 999  // 999 is blank according to docs
replace voted = . if voted > 2  // NIU/missing

* Create voting participation variable 
gen voted_bin = .
replace voted_bin = 1 if voted == 2  // Voted
replace voted_bin = 0 if voted == 1  // Did not vote
label define voted_lbl 1 "Voted" 0 "Did not vote"
label values voted_bin voted_lbl

* Create variable for "registered but did not vote" using voreg and voted variables
gen reg_novote = .
replace reg_novote = 1 if voreg == 1 & voted == 1  // Registered but did not vote
replace reg_novote = 0 if voreg == 1 & voted == 2  // Registered and voted
replace reg_novote = . if voreg != 1               // Not registered or NIU
label define reg_novote_lbl 1 "Registered, did not vote" 0 "Registered and voted"
label values reg_novote reg_novote_lbl

* Create Asian indicator variable
gen asian = 0
replace asian = 1 if inlist(race, 650, 651, 803, 806, 808, 811, 812, 813, 814, 818, 819)
label variable asian "Asian descent"
label define asian_lbl 0 "Non-Asian" 1 "Asian"
label values asian asian_lbl

* Create race categories
gen race_cat = .
replace race_cat = 1 if race == 100                 // White
replace race_cat = 2 if race == 200                 // Black
replace race_cat = 3 if asian == 1                  // Asian (created below)
replace race_cat = 4 if race == 300                 // American Indian/Aleut/Eskimo
replace race_cat = 5 if race == 700                 // Other (single) race
replace race_cat = 6 if inrange(race, 801, 820)     // Mixed race categories
label define race_cat_lbl 1 "White" 2 "Black" 3 "Asian" 4 "Native American" 5 "Other" 6 "Mixed/Other"
label values race_cat race_cat_lbl

* Create citizen categories
gen citizen_cat = .
replace citizen_cat = 1 if citizen == 1  // Born in US
replace citizen_cat = 2 if citizen == 2  // Born in US territories
replace citizen_cat = 3 if citizen == 3  // Born abroad of American parents
replace citizen_cat = 4 if citizen == 4  // Naturalized citizen
replace citizen_cat = 5 if citizen == 5  // Not a citizen
label define citizen_lbl 1 "Born in US" 2 "Born in US territories" 3 "Born abroad of US parents" 4 "Naturalized" 5 "Not citizen"
label values citizen_cat citizen_lbl

* Recode Hispanic origin
gen hispanic = (hispan > 0 & hispan < 9)
label variable hispanic "Hispanic origin"

* Create income variables
gen ln_income = ln(inctot)
label variable ln_income "Log of total personal income"

gen ln_incwage = ln(incwage)
label variable ln_incwage "Log of wage and salary income"

* Create age categories
gen age_cat = .
replace age_cat = 1 if age < 30
replace age_cat = 2 if age >= 30 & age < 45
replace age_cat = 3 if age >= 45 & age < 60
replace age_cat = 4 if age >= 60
label define age_lbl 1 "18-29" 2 "30-44" 3 "45-59" 4 "60+"
label values age_cat age_lbl

* Create education categories based on EDUC variable
gen educ_cat = .
replace educ_cat = 1 if inrange(educ, 0, 70)   // Less than high school (through 11th grade)
replace educ_cat = 2 if inrange(educ, 71, 73)  // 12th grade but no diploma or unclear
replace educ_cat = 3 if educ == 73             // High school diploma or equivalent
replace educ_cat = 4 if inrange(educ, 80, 92)  // Some college or Associate's degree
replace educ_cat = 5 if inrange(educ, 100, 125) // Bachelor's degree or higher
label define educ_lbl 1 "Less than High School" 2 "12th grade, no diploma" 3 "High School diploma" 4 "Some college/Assoc." 5 "Bachelor's+"
label values educ_cat educ_lbl

* Create variable for length of time in the US for immigrants
gen years_in_us = year - yrimmig if yrimmig > 0 & yrimmig < 9999
label variable years_in_us "Years lived in the US"

* Create immigration duration categories
gen immig_cat = .
replace immig_cat = 1 if years_in_us <= 5
replace immig_cat = 2 if years_in_us > 5 & years_in_us <= 10
replace immig_cat = 3 if years_in_us > 10 & years_in_us <= 20
replace immig_cat = 4 if years_in_us > 20
label define immig_lbl 1 "≤5 years" 2 "6-10 years" 3 "11-20 years" 4 "20+ years"
label values immig_cat immig_lbl

* Generate year categories for trend analysis
gen year_cat = .
replace year_cat = 1 if year >= 1962 & year < 1980
replace year_cat = 2 if year >= 1980 & year < 2000
replace year_cat = 3 if year >= 2000 & year < 2020
replace year_cat = 4 if year >= 2020
label define year_lbl 1 "1962-1979" 2 "1980-1999" 3 "2000-2019" 4 "2020+"
label values year_cat year_lbl

* Create interaction terms
gen asian_ln_income = asian * ln_income
gen asian_naturalized = asian * (citizen_cat == 4)
gen asian_years_in_us = asian * years_in_us

* Save the cleaned and processed data
save "cleaned_final_data.dta", replace

********************************************************************************
* DESCRIPTIVE STATISTICS
********************************************************************************

* Load data
use "cleaned_final_data.dta", clear

* Overall summary statistics
eststo clear
eststo: estpost summarize voted_bin asian ln_income age citizen_cat years_in_us educ_cat

* Summary statistics by racial group
eststo: estpost summarize voted_bin ln_income age if race_cat == 1
eststo: estpost summarize voted_bin ln_income age if race_cat == 2
eststo: estpost summarize voted_bin ln_income age if race_cat == 3

esttab using "summary_stats.tex", cells("mean sd min max count") ///
  title("Summary Statistics") label booktabs replace

********************************************************************************
* DATA VISUALIZATIONS
********************************************************************************

* Voter participation rates over time by racial group
* Only include presidential election years
preserve
* Generate indicator for presidential election years
gen preselect = inlist(year, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024)
* Restrict to citizens (citizen = 1-4 are eligible to vote)
collapse (mean) voted_rate=voted_bin (count) n=voted_bin if citizen <= 4 & preselect == 1, by(year race_cat)
twoway (connected voted_rate year if race_cat == 1) ///
       (connected voted_rate year if race_cat == 2) ///
       (connected voted_rate year if race_cat == 3) ///
       , title("Voter Participation Rates in Presidential Elections") ///
       subtitle("U.S. Citizens by Race") ///
       ytitle("Voter Participation Rate") xtitle("(Election) Year") ///
       legend(order(1 "White" 2 "Black" 3 "Asian")) ///
       ylabel(0(0.1)1) xtitle(, size(small)) ///
       xlabel(1994(2)2022, angle(45)) ///
       xscale(range(1994 2022))
graph export "voting_rates_by_race.eps", replace
restore

* Voter participation by immigration status and race
preserve
collapse (mean) voted_rate=voted_bin (count) n=voted_bin, by(citizen_cat race_cat)
graph bar voted_rate, over(citizen_cat) over(race_cat) ///
      title("Voter Participation by Citizenship Status and Race") ///
      ytitle("Voter Participation Rate") ///
      bar(1, color(navy)) asyvars
graph export "voting_by_citizenship.eps", replace
restore

* Asian voter participation by years in the US
preserve
* Filter the data to the subset
keep if asian == 1 & citizen_cat == 4
* Collapse on this filtered data
collapse (mean) voted_rate=voted_bin (count) n=voted_bin, by(immig_cat)
graph bar voted_rate, over(immig_cat) ///
      title("Asian Voter Participation by Years in the US") ///
      subtitle("Naturalized Citizens Only") ///
      ytitle("Voter Participation Rate") ///
      bar(1, color(navy))
graph export "asian_voting_by_duration.eps", replace
restore

* Voter participation by education and race
preserve
collapse (mean) voted_rate=voted_bin (count) n=voted_bin, by(educ_cat race_cat)
graph bar voted_rate, over(educ_cat) over(race_cat) ///
      title("Voter Participation by Education and Race") ///
      ytitle("Voter Participation Rate") ///
      bar(1, color(navy)) asyvars
graph export "voting_by_education.eps", replace
restore

* Changes over time for Asian Americans
preserve
* Filter the data to Asian Americans only
keep if asian == 1
* Collapse on this filtered data
collapse (mean) voted_rate=voted_bin (count) n=voted_bin, by(year_cat)
graph bar voted_rate, over(year_cat) ///
      title("Asian American Voter Participation Over Time") ///
      ytitle("Voter Participation Rate") ///
      bar(1, color(navy))
graph export "asian_voting_trends.eps", replace
restore

********************************************************************************
* REGRESSION ANALYSIS WITH TIME FIXED EFFECTS
********************************************************************************

* Base models with time fixed effects
* Model 1: Race only
eststo clear
eststo: probit voted_bin i.race_cat i.year if citizen_cat <= 4

* Model 2: Race and basic demographics
eststo: probit voted_bin i.race_cat i.sex i.age_cat i.year if citizen_cat <= 4

* Model 3: Add citizenship status
eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.year if citizen_cat <= 4

/* Model 4: Add socioeconomic factors
eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat ln_income i.educ_cat i.year if citizen_cat <= 4*/

/* Model 5: Full model with interaction terms
eststo: probit voted_bin i.race_cat##c.ln_income i.race_cat##i.citizen_cat i.sex i.age_cat i.educ_cat i.year if citizen_cat <= 4*/
// We found income variable was not aviliable in the sample with voting data non-missing.

* Export results to LaTeX table
esttab using "regression_results_timefix.tex", ///
  b(3) se(3) star(* 0.1 ** 0.05 *** 0.01) ///
  title("Probit Regression Models of Voter Participation with Time Fixed Effects") ///
  mtitles("Race Only" "Demographics" "Citizenship") ///
  label booktabs replace ///
  indicate("Year FE = *.year") // Show "Yes" for time fixed effects instead of listing all coefficients

** Models focused on Asian Americans with time fixed effects
* Model 6: Asian subsample - basic demographics
eststo clear
eststo: probit voted_bin i.sex i.age_cat i.year if asian == 1 & citizen_cat <= 4

* Model 7: Asian subsample - add citizenship
eststo: probit voted_bin i.sex i.age_cat i.citizen_cat i.year if asian == 1 & citizen_cat <= 4

/* Model 8: Asian subsample - add income and education
eststo: probit voted_bin i.sex i.age_cat i.citizen_cat ln_income i.educ_cat i.year if asian == 1 & citizen_cat <= 4*/ // We dropped the model with ln_income due to missing data

* Model 9: Asian subsample - add years in US for immigrants
* Focus on naturalized citizens (citizen_cat == 4) to analyze immigration effects
eststo: probit voted_bin i.sex i.age_cat i.educ_cat i.immig_cat i.year if asian == 1 & citizen_cat == 4

* Model 10: Compare foreign-born Asians to foreign-born non-Asians
eststo: probit voted_bin asian i.sex i.age_cat i.educ_cat i.immig_cat i.year if citizen_cat == 4

* Export results to LaTeX table
esttab using "asian_regression_results_timefix.tex", b(3) se(3) ///
  title("Probit Regression Models of Asian American Voter Participation with Time Fixed Effects") ///
  star(* 0.1 ** 0.05 *** 0.01) ///
  mtitles("Demographics" "Citizenship" "Years in US" "Naturalized Citizens") ///
  label booktabs replace ///
  indicate("Year FE = *.year") // Indicate time fixed effects

/* Trend analysis over time
forvalues y = 1980(10)2020 {
    local y2 = `y' + 9
    eststo clear
    eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat if year >= `y' & year <= `y2' & citizen_cat <= 4
    esttab using "regression_`y'_`y2'.tex", b(3) se(3) ///
      title("Voter Participation Model (`y'-`y2')") ///
      label booktabs replace
}*/ // We did not include these tables in the report due to missing data

********************************************************************************
* ROBUSTNESS CHECKS WITH TIME FIXED EFFECTS
********************************************************************************

* Alternative model specifications
* Logit model instead of probit
eststo clear
eststo: logit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat i.year if citizen_cat <= 4

* Linear probability model
eststo: regress voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat i.year if citizen_cat <= 4

* Add presidential election indicator
gen pres_election = inlist(year, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024)
eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat pres_election if citizen_cat <= 4

esttab using "robustness_checks_timefix.tex", b(3) se(3) r2 ///
  title("Robustness Check: Alternative Model Specifications with Time Controls") ///
  mtitles("Logit" "LPM" "Pres Election") ///
  label booktabs replace ///
  indicate("Year FE = *.year")

********************************************************************************
* ADDITIONAL ANALYSIS: INTERACTION WITH TIME PERIODS
********************************************************************************

* Create election cycle indicators (presidential vs midterm)
gen election_type = .
replace election_type = 1 if pres_election == 1  // Presidential election
replace election_type = 0 if pres_election == 0  // Midterm election
label define elec_lbl 1 "Presidential" 0 "Midterm"
label values election_type elec_lbl

* Analyze whether race effects differ between presidential and midterm elections
eststo clear
eststo: probit voted_bin i.race_cat##i.election_type i.sex i.age_cat i.citizen_cat i.educ_cat if citizen_cat <= 4

* Analyze changes in Asian American voting over time
eststo: probit voted_bin i.race_cat##i.year_cat i.sex i.age_cat i.citizen_cat i.educ_cat if citizen_cat <= 4

* For naturalization effects over time
eststo: probit voted_bin i.citizen_cat##i.year_cat i.race_cat i.sex i.age_cat i.educ_cat if citizen_cat <= 4

esttab using "time_interactions.tex", b(3) se(3) ///
  title("Time-Related Interactions in Voting Participation") ///
  mtitles("Election Type" "Race x Time" "Citizenship x Time") ///
  label booktabs replace

* Subsample analysis
* By education level
forvalues e = 1/5 {
    eststo clear
    eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat if educ_cat == `e' & citizen_cat <= 4
    
    local educ_title ""
    if `e' == 1 local educ_title "Less than High School"
    if `e' == 2 local educ_title "12th grade, no diploma"
    if `e' == 3 local educ_title "High school diploma"
    if `e' == 4 local educ_title "Some college/Assoc."
    if `e' == 5 local educ_title "Bachelor's+"
    
    esttab using "educ_subsample_`e'.tex", b(3) se(3) ///
      title("Voter Participation Model (`educ_title')") ///
      label booktabs replace
}

* By time period
eststo clear
eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat if year_cat == 1 & citizen_cat <= 4

eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat if year_cat == 2 & citizen_cat <= 4

eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat if year_cat == 3 & citizen_cat <= 4

eststo: probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat if year_cat == 4 & citizen_cat <= 4

esttab using "period_analysis.tex", b(3) se(3) ///
  title("Voter Participation Across Time Periods") ///
  mtitles("1980-1999" "2000-2019" "2020+") ///
  label booktabs replace


********************************************************************************
* TIME-FIXED EFFECT AND ADDITIONAL PLOTS
********************************************************************************

* Visualize year fixed effects from main model
* Run the main model with time fixed effects and store results
quietly probit voted_bin i.race_cat i.sex i.age_cat i.citizen_cat i.educ_cat i.year if citizen_cat <= 4
* Store the results
estimates store main_model

* Extract and plot the year fixed effects
margins, dydx(i.year) post
marginsplot, horizontal xline(0) recast(scatter) ///
    title("Year Fixed Effects on Voting Probability") ///
    subtitle("Full Model with Race, Demographics, Citizenship") ///
    note("Marginal effects with 95% confidence intervals") ///
    xlabel(-.15(.05).15) xtitle("Average Marginal Effect on Probability of Voting") ///
    ysize(8) xsize(6) graphregion(color(white)) bgcolor(white)
graph export "year_fixed_effects.eps", replace

/* Time trends by racial group (improved version with confidence intervals)
preserve
keep if citizen_cat <= 4 & inlist(race_cat, 1, 2, 3) // Keep only White, Black, Asian citizens
* Use 4-year bins for smoother visualization (or adjust as needed)
gen year_bin = floor(year/4)*4
collapse (mean) voted_rate=voted_bin (semean) se_voted=voted_bin (count) n=voted_bin, by(year_bin race_cat)

* Calculate confidence intervals (95%)
gen voted_rate_lb = voted_rate - 1.96*se_voted
gen voted_rate_ub = voted_rate + 1.96*se_voted

* Plot with confidence intervals
twoway (connected voted_rate year_bin if race_cat == 1, lcolor(navy) mcolor(navy) lwidth(medthick)) ///
       (rarea voted_rate_lb voted_rate_ub year_bin if race_cat == 1, color(navy%15) lcolor(%0)) ///
       (connected voted_rate year_bin if race_cat == 2, lcolor(maroon) mcolor(maroon) lwidth(medthick)) ///
       (rarea voted_rate_lb voted_rate_ub year_bin if race_cat == 2, color(maroon%15) lcolor(%0)) ///
       (connected voted_rate year_bin if race_cat == 3, lcolor(forest_green) mcolor(forest_green) lwidth(medthick)) ///
       (rarea voted_rate_lb voted_rate_ub year_bin if race_cat == 3, color(forest_green%15) lcolor(%0)) ///
       , title("Voter Participation Rates Over Time by Race", size(medium)) ///
       subtitle("U.S. Citizens Only", size(small)) ///
       ytitle("Voter Participation Rate") xtitle("Year") ///
       legend(order(1 "White" 3 "Black" 5 "Asian") cols(3) region(lcolor(none))) ///
       ylabel(0.3(0.1)0.8, angle(horizontal)) xtitle(, size(small)) ///
       xlabel(1988(8)2024, angle(45)) ///
       ysize(5) xsize(8) ///
       graphregion(color(white)) bgcolor(white)
graph export "voting_rates_by_race_ci.eps", replace
restore*/ // A plot we did not include in the report, but it stll looks good as "voting_rates_by_race.eps"

* Trend analysis with voting gap
* Calculate voting gaps over time
preserve
* Keep only citizens and White/Asian for comparison
keep if citizen_cat <= 4 & (race_cat == 1 | race_cat == 3)
* Collapse by race and year
collapse (mean) voted_bin (count) n=voted_bin, by(year race_cat)
* Reshape to wide format
reshape wide voted_bin n, i(year) j(race_cat)
* Calculate voting gap
gen voting_gap = voted_bin1 - voted_bin3

* Generate fitted trend
reg voting_gap year
predict gap_fitted

* Create visualization of gap with trend
twoway (scatter voting_gap year, mcolor(navy) msymbol(circle_hollow)) ///
    (line gap_fitted year, lcolor(maroon) lwidth(medthick) lpattern(solid)) ///
    , title("White-Asian Voting Participation Gap Over Time", size(medium)) ///
    xtitle("Year", size(small)) ytitle("Percentage Point Gap", size(small)) ///
    note("Positive values indicate higher White participation relative to Asian", size(vsmall)) ///
    graphregion(color(white)) bgcolor(white) ///
    ylabel(0(.05)0.25, angle(horizontal)) ///
    xlabel(1992(4)2024, angle(45))
graph export "voting_gap_trend.eps", replace
restore

** Naturalization duration and voting probability interactive visualization
preserve
* Keep naturalized citizens
keep if citizen_cat == 4 & !missing(years_in_us)

* Create a continuous visualization for years in US vs voting probability
* Bin years in US into 5-year groups for smoother visualization
gen years_in_us_bin = 5 * floor(years_in_us/5)

* Collapse by race and years in US bin
collapse (mean) voted_rate=voted_bin (count) n=voted_bin, by(years_in_us_bin race_cat)

* Keep only White, Black, and Asian for clarity
keep if inlist(race_cat, 1, 2, 3) & n >= 30 // Ensure reasonable sample size

* Create spikeplot with connected lines
twoway (connected voted_rate years_in_us_bin if race_cat == 1, lcolor(navy) mcolor(navy) lwidth(medthick)) ///
       (connected voted_rate years_in_us_bin if race_cat == 2, lcolor(maroon) mcolor(maroon) lwidth(medthick)) ///
       (connected voted_rate years_in_us_bin if race_cat == 3, lcolor(forest_green) mcolor(forest_green) lwidth(medthick)) ///
       , title("Voting Rates by Years in US for Naturalized Citizens", size(medium)) ///
       subtitle("By Racial Group", size(small)) ///
       xtitle("Years in United States", size(small)) ytitle("Voter Participation Rate", size(small)) ///
       legend(order(1 "White" 2 "Black" 3 "Asian") rows(1) region(lcolor(none))) ///
       graphregion(color(white)) bgcolor(white) ///
       ylabel(0.3(0.1)0.8, angle(horizontal)) ///
       xlabel(0(10)50)
graph export "naturalization_duration_voting.eps", replace
restore

* Close the log file 

log close

exit
