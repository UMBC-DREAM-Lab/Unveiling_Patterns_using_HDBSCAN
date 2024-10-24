# Unveiling_Patterns_using_HDBSCAN
For the paper titled "Ransomware Evolution: Unveiling Patterns Using HDBSCAN"


This research presents an innovative approach to enhancing ransomware detection by leveraging Windows API calls and PE header information to develop precise signatures capable of identifying ransomware families. Our methodology introduces a novel application of hierarchical clustering using the HDBSCAN algorithm, in conjunction with the Jaccard similarity metric, to cluster ransomware into discrete families and generate corresponding signatures. This technique, to our knowledge, marks a pioneering effort in applying hierarchical density-based clustering to over 1.1 million malicious samples, specifically focusing on ransomware and using the clusters to automatically generate signatures.

We show that identifying unique Windows API function patterns within these clusters enables the differentiation and characterization of various ransomware families. Furthermore, we conducted a case study focusing on the distinctive function combinations within prominent ransomware families such as GandCrab, WannaCry, Cerber, Gotango, and CryptXXX, unveiling unique behaviors and API function usage patterns. Our scalable implementation demonstrates the ability to efficiently cluster large volumes of malicious files and automatically generate robust, actionable function signatures for each. Validation of these signatures on an independent malware dataset yielded a precision rate of 98.34\% and specificity rate of 99.72\%, affirming their 
effectiveness in detecting known ransomware families with minimal error. These findings underscore the potential of our methodology in bolstering cybersecurity defenses against the evolving landscape of ransomware threats.

#Contact
email: prajnab1@umbc.edu
