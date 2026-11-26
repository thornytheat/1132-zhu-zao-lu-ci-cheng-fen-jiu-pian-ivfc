from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
from math import ceil, isfinite
from .contracts import CorrectionBatch


def bench_units(row, increments_per_kg, step):
    if increments_per_kg<=0 or step<1:
        raise ValueError("SCALE_STEP_INVALID")
    increment=Decimal(str(increments_per_kg))*Decimal(step)
    goal=Decimal(str(row.deficit))*Decimal(str(row.mass_kg))
    allowed=Decimal(str(row.space))*Decimal(str(row.mass_kg))
    needed=int((goal/increment).to_integral_value(rounding=ROUND_CEILING))*step
    maximum=int((allowed/increment).to_integral_value(rounding=ROUND_DOWN))*step
    return needed, maximum


def correction_plan(records, config, total_allowance, samples=None, **joint_options):
    """Plan corrections.

    With joint residual context this dispatches to the whole-furnace planner.
    Without it, each heat is still allocated atomically when it contains several
    valid correction lines; single lines retain the original reasons exactly.
    """
    joint_samples=joint_options.pop("joint_samples",None)
    if samples is None:
        samples=joint_samples
    if samples is not None or joint_options.get("samples_by_grade") is not None:
        if samples is None:
            samples=joint_options.pop("samples",None)
        return whole_heat_plan(records,config,total_allowance,samples=samples,**joint_options)
    if total_allowance<0:
        raise ValueError("NEGATIVE_BATCH_ALLOWANCE")
    key_counts=Counter((row.heat,row.element) for row in records)
    ordered=sorted(records,key=lambda x:(-x.priority,x.heat,x.element))
    eligibility={}
    packages=defaultdict(list)
    for row in ordered:
        approved=0
        required=0
        if key_counts[(row.heat,row.element)]>1:
            state="重复纠偏申请"
        elif row.state!="成分下界偏低":
            state=row.state
        elif row.element not in config:
            state="未配置纠偏称"
        else:
            rules=config[row.element]
            required,safe=bench_units(row,rules["gain"],rules["step"])
            if required>safe:
                state="最小称量超过安全余量"
            else:
                state="__ELIGIBLE__"
                packages[row.heat].append((row,required))
        eligibility[id(row)]=(state,required)
    spent=0
    used=Counter()
    heat_order=list(dict.fromkeys(row.heat for row in ordered))
    row_decisions={}
    for heat in heat_order:
        items=packages.get(heat,[])
        if not items:
            continue
        total=sum(required for _row,required in items)
        element_totals=Counter()
        for row,required in items:
            element_totals[row.element]+=required
        element_blocked=any(used[element]+amount>config[element]["limit"]
                            for element,amount in element_totals.items())
        capacity_blocked=spent+total>total_allowance
        if element_blocked or capacity_blocked:
            multi=len(items)>1
            reason="整炉排队" if multi and capacity_blocked else "纠偏称量额度不足"
            for row,required in items:
                row_decisions[id(row)]=(0,reason)
        else:
            for row,required in items:
                row_decisions[id(row)]=(required,"已批准复验" if required else "无需纠偏")
            spent+=total
            for element,amount in element_totals.items():
                used[element]+=amount
    ledger=[]
    for row in ordered:
        initial_state,required=eligibility[id(row)]
        if initial_state=="__ELIGIBLE__":
            approved,state=row_decisions[id(row)]
        else:
            approved,state=0,initial_state
        ledger.append(dict(heat=row.heat,element=row.element,grade=row.grade,
                           approved_units=approved,reason=state))
    return ledger,{element:amount for element,amount in used.items() if amount}


def _residual_map(sample, elements):
    if hasattr(sample, "residuals"):
        values=sample.residuals
    elif isinstance(sample, dict) and isinstance(sample.get("residuals"), dict):
        values=sample["residuals"]
    elif isinstance(sample, dict):
        values=sample
    else:
        values={element:getattr(sample, element, None) for element in elements}
    return {element:float(values[element]) for element in elements
            if element in values and values[element] is not None and isfinite(float(values[element]))}


def evaluate_heat_plan(forecast, units, bounds, samples, rules=None, mass=1.0, radii=None):
    """Return one hit flag per complete empirical furnace residual vector."""
    elements=list(bounds)
    hits=[]
    if not isinstance(mass,dict):
        try:
            mass=float(mass)
        except (TypeError, ValueError):
            mass=1.0
        if not isfinite(mass) or mass<=0:
            mass=1.0
    for sample in samples:
        residual=_residual_map(sample,elements)
        if any(element not in residual for element in elements):
            hits.append(False)
            continue
        ok=True
        for element in elements:
            rule=(rules or {}).get(element,{})
            gain=rule.get("gain",0) if isinstance(rule,dict) else 0
            if isinstance(mass,dict):
                element_mass=float(mass.get(element,1.0))
            else:
                element_mass=float(mass)
            if not isfinite(element_mass) or element_mass<=0:
                element_mass=1.0
            correction=float(gain)*float(units.get(element,0))/element_mass
            center=float(forecast[element])+correction+residual[element]
            low,high=bounds[element]
            radius=radii.get(element,0.0) if isinstance(radii,dict) else (radii or 0.0)
            if center-float(radius)<float(low) or center+float(radius)>float(high):
                ok=False
                break
        hits.append(ok)
    return hits


def joint_hit_rate(*args, **kwargs):
    hits=evaluate_heat_plan(*args,**kwargs)
    return sum(hits)/len(hits) if hits else 0.0


heat_hit_rate=joint_hit_rate
whole_heat_hit_rate=joint_hit_rate


def minimum_heat_plan(rows, rules, *, forecast, bounds, samples,
                      confidence=.9, coverage=None,
                      minimum_support=4, required_support=None,
                      distances=None, support_distance=7.5,
                      radii=None, **_options):
    """Find the least-reservation discrete dose vector hitting joint confidence."""
    if coverage is not None:
        confidence=coverage
    if required_support is not None:
        minimum_support=required_support
    heat=rows[0].heat if rows else None
    grade=rows[0].grade if rows else None
    elements=sorted({row.element for row in rows})
    result=dict(heat=heat,grade=grade,status="suspended",reason=None,reasons={},
                units={element:0 for element in elements},total=0,hits=0,
                support=len(samples),confidence=0.0)
    if not 0<confidence<=1:
        raise ValueError("INVALID_CONFIDENCE")
    for row in rows:
        rule=rules.get(row.element)
        if isinstance(rule,dict):
            gain=rule.get("gain",0)
            step=rule.get("step",1)
            if not isfinite(float(gain)) or float(gain)<=0 or int(step)<1:
                raise ValueError("SCALE_STEP_INVALID")
    if support_distance is not None and distances is not None:
        distance_map=distances if isinstance(distances,dict) else {}
        for element in elements:
            distance=distance_map.get(element)
            if distance is not None and float(distance)>support_distance:
                result.update(reason="历史工况外复验",
                              reasons={item.element:"历史工况外复验" for item in rows})
                return result
    complete=[sample for sample in samples
              if all(element in _residual_map(sample,elements) for element in elements)]
    result["support"]=len(complete)
    if len(complete)<minimum_support:
        reason="完整样本不足"
        result.update(reason=reason,reasons={item.element:reason for item in rows})
        return result

    choices={}
    required_safe={}
    for row in rows:
        element=row.element
        if row.state=="重复纠偏申请":
            result.update(reason="重复纠偏申请",reasons={element:"重复纠偏申请"})
            return result
        if row.state not in ("成分下界偏低","合格区间"):
            result.update(reason=row.state,reasons={element:row.state})
            return result
        rule=rules.get(element)
        if not isinstance(rule,dict):
            if row.state=="合格区间":
                required_safe[element]=(0,0)
                choices[element]=[0]
                continue
            reason="未配置纠偏称"
            result.update(reason=reason,reasons={element:reason})
            return result
        gain=rule.get("gain",0)
        step=rule.get("step",1)
        if not isfinite(float(gain)) or float(gain)<=0 or int(step)<1:
            raise ValueError("SCALE_STEP_INVALID")
        required,safe=bench_units(row,gain,step)
        required_safe[element]=(required,safe)
        limit=rule.get("limit")
        maximum=safe if limit is None else min(int(limit),safe)
        choices[element]=list(range(0,maximum+1,int(step)))
        if not choices[element]:
            choices[element]=[0]

    need=ceil(confidence*len(complete)-1e-12)
    best=None

    def search(index, units, total):
        nonlocal best
        if best is not None and total>best["total"]:
            return
        if index==len(elements):
            hit_flags=evaluate_heat_plan(
                forecast,units,bounds,complete,rules=rules,
                mass={row.element:row.mass_kg for row in rows},
                radii=radii)
            hits=sum(hit_flags)
            if hits>=need and (best is None or total<best["total"] or
                               (total==best["total"] and
                                [units[e] for e in elements]<[best["units"][e] for e in elements])):
                best=dict(units=dict(units),total=total,hits=hits,
                          confidence=hits/len(complete))
            return
        element=elements[index]
        for amount in choices[element]:
            units[element]=amount
            search(index+1,units,total+amount)
        units[element]=0

    search(0,{},0)
    if best is None:
        if any(required>safe for required,safe in required_safe.values()):
            reason="最小称量超过安全余量"
        else:
            reason="整炉置信方案不足"
        result.update(reason=reason,reasons={item.element:reason for item in rows})
        return result
    result.update(status="ready",reason="已批准复验",units=best["units"],total=best["total"],
                  hits=best["hits"],confidence=best["confidence"])
    return result


minimum_reservation=minimum_heat_plan
whole_heat_reservation=minimum_heat_plan


def _context_lookup(value, heat, element, default=None):
    if value is None:
        return default
    if isinstance(value,dict):
        if (heat,element) in value:
            return value[heat,element]
        if heat in value and isinstance(value[heat],dict):
            return value[heat].get(element,default)
        if element in value and not isinstance(value[element],dict):
            return value[element]
    return default


def whole_heat_plan(records, config, total_allowance, samples=None, *,
                    samples_by_grade=None, forecasts=None, forecast=None,
                    bounds=None, bound=None, grade_bounds=None,
                    distances=None, radii=None,
                    confidence=.9, coverage=None,
                    minimum_support=4, required_support=None,
                    min_samples=None,
                    support_distance=7.5, distance_limit=None, **_options):
    """Plan and atomically allocate whole-furnace bench reservations."""
    if coverage is not None:
        confidence=coverage
    if required_support is not None:
        minimum_support=required_support
    if min_samples is not None:
        minimum_support=min_samples
    if forecasts is None:
        forecasts=forecast
    if bounds is None:
        bounds=bound
    if samples is None:
        samples=(_options.pop("residual_vectors",None)
                 or _options.pop("joint_samples",None)
                 or _options.pop("residuals",None))
    if total_allowance<0:
        raise ValueError("NEGATIVE_BATCH_ALLOWANCE")
    if distance_limit is not None:
        support_distance=distance_limit
    key_counts=Counter((row.heat,row.element) for row in records)
    ordered=sorted(records,key=lambda x:(-x.priority,x.heat,x.element))
    heats=list(dict.fromkeys(row.heat for row in ordered))
    grouped=defaultdict(list)
    for row in ordered:
        grouped[row.heat].append(row)

    plans={}
    base_reason={}
    for row in ordered:
        if key_counts[(row.heat,row.element)]>1:
            base_reason[id(row)]="重复纠偏申请"
        elif row.state=="合格区间":
            base_reason[id(row)]="__JOINT__"
        elif row.state!="成分下界偏低":
            base_reason[id(row)]=row.state
        elif row.element not in config:
            base_reason[id(row)]="未配置纠偏称"
        else:
            base_reason[id(row)]="__JOINT__"

    for heat in heats:
        rows=grouped[heat]
        grades={row.grade for row in rows}
        if len(grades)!=1:
            plans[heat]=dict(status="suspended",reason="待核产品牌号",units={},
                            total=0,reasons={row.element:"待核产品牌号" for row in rows})
            continue
        grade=next(iter(grades))
        heat_forecast={}
        heat_bounds={}
        heat_distances={}
        heat_radii={}
        context_ready=True
        for row in rows:
            predicted=_context_lookup(forecasts,heat,row.element)
            if predicted is None:
                context_ready=False
            heat_forecast[row.element]=predicted
            bound=_context_lookup(bounds,heat,row.element)
            if bound is None and grade_bounds is not None:
                bound=grade_bounds.get(grade,{}).get(row.element)
            if bound is None:
                context_ready=False
            heat_bounds[row.element]=bound
            distance=_context_lookup(distances,heat,row.element)
            if distance is not None:
                heat_distances[row.element]=distance
            radius=_context_lookup(radii,heat,row.element)
            if radius is not None:
                heat_radii[row.element]=radius
        if not context_ready:
            # Direct deterministic callers without forecast context retain old
            # per-line allocation semantics through correction_plan().
            plans[heat]=None
            continue
        grade_samples=samples
        if samples_by_grade is not None:
            grade_samples=samples_by_grade.get(grade,samples or [])
        plans[heat]=minimum_heat_plan(
            rows,config,forecast=heat_forecast,bounds=heat_bounds,
            samples=grade_samples or [],confidence=confidence,
            minimum_support=minimum_support,
            distances=heat_distances or None,
            support_distance=support_distance,
            radii=heat_radii or None)

    spent=0
    used=Counter()
    if all(plan is None for plan in plans.values()):
        return correction_plan(records,config,total_allowance)
    allocation={}
    priority=lambda heat:max(row.priority for row in grouped[heat])
    for heat in sorted(heats,key=lambda heat:(-priority(heat),heat)):
        plan=plans[heat]
        if plan is None or plan.get("status")!="ready":
            continue
        units=plan["units"]
        total=plan["total"]
        element_totals=Counter()
        for row in grouped[heat]:
            element_totals[row.element]+=int(units.get(row.element,0))
        if spent+total>total_allowance:
            allocation[heat]=("整炉排队",{row.element:0 for row in grouped[heat]})
            continue
        limits_ok=True
        for element,amount in element_totals.items():
            limit=config.get(element,{}).get("limit")
            if limit is not None and (amount>limit or used[element]+amount>limit):
                limits_ok=False
                break
        if spent+total>total_allowance or not limits_ok:
            reason="整炉排队" if spent+total>total_allowance else "纠偏称量额度不足"
            allocation[heat]=(reason,{row.element:0 for row in grouped[heat]})
            continue
        allocation[heat]=("__APPROVED__",dict(units))
        spent+=total
        for element,amount in element_totals.items():
            used[element]+=amount

    ledger=[]
    for row in ordered:
        initial=base_reason[id(row)]
        approved=0
        if initial!="__JOINT__":
            reason=initial
        else:
            plan=plans.get(row.heat)
            decision=allocation.get(row.heat)
            if plan is None:
                # Context-free fallback: preserve the original deterministic rule.
                required,safe=bench_units(row,config[row.element]["gain"],config[row.element]["step"])
                if required>safe:
                    reason="最小称量超过安全余量"
                elif decision is None:
                    reason="纠偏称量额度不足"
                else:
                    reason=row.state
            elif decision is not None:
                allocation_reason,units=decision
                if allocation_reason=="__APPROVED__":
                    approved=int(units.get(row.element,0))
                    reason="已批准复验" if approved else ("合格区间" if row.state=="合格区间" else "无需纠偏")
                else:
                    reason=allocation_reason
            else:
                reason=plan.get("reasons",{}).get(row.element) or plan.get("reason") or row.state
        ledger.append(dict(heat=row.heat,element=row.element,grade=row.grade,
                           approved_units=approved,reason=reason))
    return ledger,{element:amount for element,amount in used.items() if amount}


def allocate_whole_heats(plans, config, total_allowance):
    """Allocate already-computed minimum heat plans atomically."""
    used=Counter()
    spent=0
    decisions={}
    for plan in sorted(plans,key=lambda item:(-item.get("priority",0),item.get("heat",""))):
        units=plan.get("units",{})
        total=int(plan.get("total",sum(units.values())))
        heat=plan["heat"]
        if plan.get("status")!="ready":
            decisions[heat]=(plan.get("reason","整炉挂起"),units,False)
            continue
        limits_ok=all(used[element]+amount<=config.get(element,{}).get("limit",amount)
                      for element,amount in units.items())
        if spent+total>total_allowance or not limits_ok:
            decisions[heat]=("整炉排队",{element:0 for element in units},False)
            continue
        spent+=total
        for element,amount in units.items():
            used[element]+=amount
        decisions[heat]=("已批准复验",units,True)
    return decisions,dict(used),spent


joint_reservation_plan=whole_heat_plan
heat_reservation_plan=whole_heat_plan
plan_whole_heat=whole_heat_plan
reserve_whole_heat=whole_heat_plan


deterministic_correction_plan=correction_plan
itemwise_correction_plan=correction_plan


joint_correction_plan=whole_heat_plan
reservation_plan=whole_heat_plan
bench_reservation_plan=whole_heat_plan


whole_furnace_plan=whole_heat_plan
furnace_reservation_plan=whole_heat_plan
batch_whole_heat=whole_heat_plan


joint_heat_plan=whole_heat_plan
whole_heat_reservation_plan=whole_heat_plan
