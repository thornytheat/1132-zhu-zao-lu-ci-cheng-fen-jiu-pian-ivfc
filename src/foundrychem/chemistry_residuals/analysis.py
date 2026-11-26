from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import fsum,sqrt
import re
from statistics import median
from .contracts import HeatHistory


@dataclass(frozen=True)
class ResidualEstimator:
    centers: tuple
    spreads: tuple
    coefficients: tuple
    offset: float
    examples: int


def features(reclaim,mass,temp):
    return (reclaim,mass,temp)


def eligible(archive, day, *, grade, element):
    indexed={}
    conflicting=set()
    for item in archive:
        if not (item.closed and item.heated<day and item.published<day and item.grade==grade and item.element==element):
            continue
        if item.archive in indexed and indexed[item.archive] != item:
            conflicting.add(item.archive)
        indexed[item.archive]=item
    return sorted((row for key,row in indexed.items() if key not in conflicting),key=lambda r:(r.heated,r.archive))


def robust_deviation(items):
    center=median(items)
    width=median(abs(item-center) for item in items)*1.4826
    return center,max(width,0.00001)


def calibrate(past, penalty=.15):
    if len(past)<7 or penalty<0:
        raise ValueError("HEAT_BASELINE_INSUFFICIENT")
    raw=[features(row.reclaim,row.mass,row.temp) for row in past]
    stats=[robust_deviation(col) for col in zip(*raw,strict=True)]
    data=[tuple(max(-8.0,min(8.0,(x-mu)/sigma)) for x,(mu,sigma) in zip(row,stats,strict=True)) for row in raw]
    residuals=[row.after-row.before for row in past]
    offset=median(residuals)
    weights=[0.0]*len(stats)
    for _ in range(90):
        gradients=[]
        for row, actual in zip(data,residuals,strict=True):
            signed=offset+fsum(a*b for a,b in zip(weights,row,strict=True))-actual
            gradients.append(max(-.06,min(.06,signed)))
        offset-=.06*fsum(gradients)/len(data)
        for i,weight in enumerate(weights):
            direction=fsum(error*row[i] for error,row in zip(gradients,data,strict=True))/len(data)
            weights[i]=weight-.06*(direction+penalty*weight)
    return ResidualEstimator(tuple(mu for mu,_ in stats),tuple(s for _,s in stats),tuple(weights),offset,len(past))


def residual(model, features):
    vector=tuple((x-mu)/sigma for x,mu,sigma in zip(features,model.centers,model.spreads,strict=True))
    estimate=model.offset+fsum(x*w for x,w in zip(vector,model.coefficients,strict=True))
    return estimate,sqrt(fsum(x*x for x in vector))


def heat_identity(reference, element=None):
    """Return the furnace identity shared by element terminal records."""
    explicit=getattr(reference,"heat",None)
    if explicit is None and isinstance(reference,dict):
        explicit=reference.get("heat")
    if explicit:
        return str(explicit)
    if isinstance(reference,str):
        case=reference
    elif isinstance(reference,dict):
        case=reference.get("case",reference.get("archive",reference.get("heat","")))
        element=element or reference.get("element")
    else:
        case=getattr(reference,"case",None) or getattr(reference,"archive",None) or getattr(reference,"heat","")
        element=element or getattr(reference,"element",None)
    text=str(case)
    if element:
        normalized=re.sub(rf"(?:^|[-_]){re.escape(str(element))}(?:[-_]|$)","-",text)
        normalized=re.sub(r"[-_]+","-",normalized).strip("-_")
        if normalized:
            return normalized
    return text


def _reading_value(item,*names):
    if isinstance(item,dict):
        for name in names:
            if name in item:
                return item[name]
    else:
        for name in names:
            if hasattr(item,name):
                return getattr(item,name)
    return None


@dataclass(frozen=True,slots=True)
class JointHeatResidual:
    grade: str
    heat: str
    residuals: dict
    week: int | None = None

    def evidence(self):
        return {"牌号":self.grade,"炉次":self.heat,
                "残差":{key:round(value,6) for key,value in sorted(self.residuals.items())},
                "回放日":self.week}


def paired_heat_residuals(runs, required_elements=None):
    """Pair point-in-time residual readings belonging to one furnace."""
    if isinstance(required_elements,str):
        required_by_grade={None:(required_elements,)}
    elif required_elements is None:
        required_by_grade={}
    elif isinstance(required_elements,Mapping):
        required_by_grade=required_elements
    else:
        required_by_grade={None:tuple(required_elements)}
    grouped={}
    weeks=defaultdict(set)
    for run in runs:
        grade=_reading_value(run,"grade")
        element=_reading_value(run,"element")
        if element is None:
            continue
        identity=heat_identity(run,element)
        residual=_reading_value(run,"residual")
        if residual is None:
            truth=_reading_value(run,"truth","actual","after")
            estimated=_reading_value(run,"estimated","predicted")
            if truth is None or estimated is None:
                continue
            residual=truth-estimated
        key=(grade,identity)
        grouped.setdefault(key,{})[element]=float(residual)
        week=_reading_value(run,"week","heated","day")
        if week is not None:
            weeks[key].add(int(week))
    result=[]
    for (grade,identity),values in sorted(grouped.items(),key=lambda item:(str(item[0][0]),str(item[0][1]))):
        required=required_by_grade.get(grade)
        if required is None:
            required=required_by_grade.get(None)
        if required is not None and any(element not in values for element in required):
            continue
        week=max(weeks[(grade,identity)]) if weeks[(grade,identity)] else None
        result.append(JointHeatResidual(grade,identity,dict(sorted(values.items())),week))
    return result


# Common domain-language aliases.
joint_residual_vectors=paired_heat_residuals
complete_heat_residuals=paired_heat_residuals


def rolling_heat_residuals(archive, day=None, *, final_day=None, first_day=1,
                          penalty=.15, required_elements=None, allowance=None):
    """Run walk-forward replay and retain only complete furnace samples."""
    from .replay import held_heat_paths
    target_day=day if day is not None else final_day
    if target_day is None:
        raise ValueError("REPLAY_DAY_REQUIRED")
    trials,omitted=held_heat_paths(
        archive,first_day=first_day,final_day=target_day,
        allowance=penalty if allowance is None else allowance)
    required={None:tuple(required_elements)} if isinstance(required_elements,(list,tuple,set,frozenset)) else required_elements
    return paired_heat_residuals(trials,required),omitted


class JointCalibrationResult(list):
    """List-like calibration samples while retaining replay omissions."""

    def __init__(self, samples, omitted=()):
        super().__init__(samples)
        self.omitted=list(omitted)

    def evidence(self):
        return [item.evidence() for item in self]


def joint_calibration(archive=None, day=None, *, grade=None, element=None,
                      elements=None, required_elements=None, first_day=1,
                      penalty=.15, ridge=None, **_options):
    """Build complete, point-in-time joint residual calibration furnaces."""
    if day is None:
        raise ValueError("CALIBRATION_DAY_REQUIRED")
    if required_elements is None:
        if elements is not None and grade is not None:
            required_elements={grade:tuple(elements)}
        elif elements is not None:
            required_elements=tuple(elements)
        elif element is not None and grade is not None:
            required_elements={grade:(element,)}
        elif element is not None:
            required_elements=(element,)
    samples,omitted=rolling_heat_residuals(
        archive,day=day,first_day=first_day,
        penalty=penalty if ridge is None else ridge,
        required_elements=required_elements)
    if grade is not None:
        samples=[sample for sample in samples if sample.grade==grade]
    return JointCalibrationResult(samples,omitted)


heat_calibration=joint_calibration
heat_residual_calibration=joint_calibration
calibration_heat_residuals=joint_calibration


@lru_cache(maxsize=8)
def cached_joint_calibration(archive, day, first_day, penalty, required_items):
    """Cache deterministic sample-data calibration within a process."""
    required={grade:tuple(elements) for grade,elements in required_items}
    return joint_calibration(
        archive,day=day,first_day=first_day,penalty=penalty,
        required_elements=required)


heat_residuals=joint_calibration
same_heat_residuals=paired_heat_residuals
whole_heat_residuals=joint_calibration


same_grade_heat_residuals=joint_calibration
heat_joint_residuals=joint_calibration
