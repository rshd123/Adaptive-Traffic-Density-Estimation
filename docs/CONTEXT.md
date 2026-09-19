# Adaptive Traffic Density Estimation for Chennai Roads

## What Problem Does It Solve in Chennai Traffic?

Chennai roads are lane-less and dominated by mixed vehicle types (cars, bikes, autos, buses, trucks all sharing the same stretch of road with no lane discipline). Most existing traffic-density systems either:

- assume clean lanes (breaks on Chennai roads), or
- count every vehicle equally (a bike and a bus "count" the same, which misrepresents actual road congestion), or
- undercount motorcycles/autos when they cluster and overlap (very common at Chennai signals/junctions).

This system directly targets these three failure points: it works without lane markings (zone-based), reflects true road-space usage via PCU weighting (a bus doesn't equal a bike), and corrects for the undercounting of clustered two-wheelers/autos via occlusion-aware detection. In plain terms — it gives a more accurate, realistic picture of how congested a Chennai road actually is, which is the necessary first step before any signal-timing or congestion-management system can work properly.

## Is It Publishable?

Yes, realistically. Here's why:

| Factor | Assessment |
|---|---|
| Novelty | Real — the specific combination (lane-independent zones + PCU-weighting + occlusion-aware counting) as a unified framework isn't done together in existing reviewed literature. |
| Scope | Appropriately narrowed — single clear contribution + ablation study is exactly what conference/journal reviewers want. |
| Validation plan | Sound — baseline comparisons + ablations + external Chennai validation gives a full experimental section. |
| Venue fit | Very publishable at IEEE conference level (similar tier to RMKMATE, ICCAIS, ICAFT, CSNT). A top-tier journal (e.g., IEEE T-ITS) would need stronger real-world deployment/larger-scale validation, but a solid conference paper is well within reach. |
| Risk | Main risk is dataset limitations (IDD-FGVD may not perfectly match Chennai composition) — mitigated by treating Chennai as external validation rather than the core training ground, which is honest and defensible. |

## Proposed Solution

We propose a YOLOv8-based adaptive traffic density estimation system tailored for heterogeneous, lane-less Chennai road traffic. Unlike existing approaches that treat all vehicles equally in density computation, our system assigns Passenger Car Unit (PCU) weights to each detected vehicle class (car, motorcycle, auto-rickshaw, bus, truck) to produce a heterogeneity-aware density score rather than a raw vehicle count. The road is divided into virtual width-wise zones/strips (rather than relying on formal lane markings, which are frequently absent or ignored on Chennai roads) to localize congestion spatially instead of producing a single aggregate value. To address the motorcycle/two-wheeler detection weakness observed across prior YOLO-based studies, the system incorporates an occlusion-correction factor and confidence-calibrated counting, reducing undercounting in dense, overlapping traffic clusters common at Chennai intersections. The resulting zone-wise, PCU-weighted density score is benchmarked against both a traditional vehicle-count baseline and classical strip-based density estimation methods, aiming to demonstrate improved accuracy and practical applicability for real-world adaptive signal control in mixed Indian traffic conditions.

**Core novelty:** Not YOLOv8 itself (vehicle detection and traffic-density estimation via YOLO are already well studied), but the combination and rigorous validation of lane-independent spatial localization + PCU-weighted heterogeneous traffic representation + occlusion-aware counting as a unified density-estimation framework for Indian urban traffic — with Chennai used as a real-world external evaluation environment.