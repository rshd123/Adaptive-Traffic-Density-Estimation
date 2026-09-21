# Adaptive Traffic Density Estimation for Chennai Roads

## What Problem Does It Solve in Chennai Traffic?

Chennai roads experience heterogeneous traffic where cars, motorcycles, auto-rickshaws, buses, and trucks share the same road space, often without clearly defined or consistently followed lane markings. Conventional traffic-density systems generally rely on raw vehicle counts or lane-based density estimation, which can provide an incomplete representation of congestion in such environments.

This system addresses two key limitations:

- **Equal vehicle counting:** A motorcycle, car, bus, and truck are treated as one vehicle even though they occupy different amounts of road space and have different effects on traffic flow.
- **Lane-dependent density estimation:** Traditional lane-based approaches are less suitable for roads where formal lane markings are absent, unclear, or not consistently followed.

The proposed system therefore estimates traffic density using **PCU (Passenger Car Unit) weighting** and **spatially localized virtual road zones**, providing a more representative measure of heterogeneous traffic congestion.

In simple terms, instead of asking:

> "How many vehicles are on the road?"

the system asks:

> "How much traffic load is present, and where on the road is that load concentrated?"

---

## Is It Publishable?

Yes, the focused formulation provides a clearer and more defensible research direction.

| Factor | Assessment |
|---|---|
| Novelty | Focuses on the combination of PCU-weighted traffic representation and lane-independent spatial density estimation |
| Scope | Clearly focused on one central research problem |
| Validation | Can be evaluated against raw vehicle counting and conventional density estimation methods |
| Dataset | DriveIndia provides a large heterogeneous Indian traffic dataset, while IDD-FGVD provides fine-grained Indian vehicle categories |
| Chennai relevance | Chennai can be used as an external real-world evaluation environment |
| Main risk | The PCU weights and density formulation must be properly justified and validated rather than chosen arbitrarily |

The research contribution should not be presented as simply using YOLOv8, since YOLO-based vehicle detection and traffic-density estimation are already well studied.

---

## Proposed Solution

We propose a **YOLOv8-based PCU-weighted spatial traffic density estimation system** designed for heterogeneous and weakly lane-disciplined urban traffic conditions.

YOLOv8 is used to detect and classify relevant vehicle categories such as:

- Car
- Motorcycle
- Auto-rickshaw
- Bus
- Truck

Instead of assigning the same contribution to every detected vehicle, the system applies **Passenger Car Unit (PCU) weights** to different vehicle classes. This produces a weighted traffic-load measure that better represents the heterogeneous composition of the road.

The road is additionally divided into **virtual width-wise spatial zones/strips** rather than relying on formal lane markings. The PCU-weighted vehicles are assigned to these zones, allowing the system to estimate not only the overall traffic density but also **where traffic is concentrated spatially**.

The resulting density can therefore be represented as:

```text
Traffic Image / Video
        ↓
      YOLOv8
        ↓
Vehicle Detection + Classification
        ↓
     PCU Weighting
        ↓
Virtual Spatial Zones
        ↓
Zone-wise PCU Density
        ↓
Overall Traffic Density