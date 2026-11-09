"""Aggregate plots. Axes are in English for portable rendering on Windows."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def finite_points(values):
    return [float("nan") if x is None else x for x in values]


BLUE, TEAL, RUST, GRAY = "#315c9a", "#137c77", "#ad572c", "#6b7280"
PALETTE = (BLUE, TEAL, RUST, "#79518d", "#354e55")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 10,
                     "figure.facecolor": "#ffffff", "axes.axisbelow": True})


def empty(ax):
    ax.text(.5, .48, "No valid observations", ha="center", va="center", color=GRAY,
            transform=ax.transAxes)
    ax.set(xticks=[], yticks=[])


def style(ax, title, n=None, ylabel=None, fraction=False):
    ax.set_title(title if n is None else f"{title}  |  n={n:,}", loc="left", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if fraction:
        ax.set_ylim(-.025, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=.18)


def counts(ax, labels, values, n, title, ylabel="Share of valid values"):
    style(ax, title, n, ylabel, fraction=True)
    if not n:
        empty(ax)
        return
    ax.bar(labels, [v/n for v in values], color=BLUE, width=.72)


def log_bins(ax, d, title, x):
    style(ax, title, d["valid"], "Share of valid values", fraction=True)
    if not d["valid"]:
        empty(ax)
        return
    # Keep ordered power-of-two bins and real gaps on the exponent axis.
    positions, shares = [], []
    positives = [r["lower"] for r in d["bins"] if r["lower"] > 0]
    origin = math.floor(math.log2(min(positives))) if positives else 0
    for r in d["bins"]:
        positions.append(math.floor(math.log2(r["lower"])) if r["lower"] else origin-1)
        shares.append(r["fraction"])
    ax.bar(positions, shares, color=[TEAL if r["lower"] else RUST for r in d["bins"]], width=.8)
    if len(positions) <= 9:
        ax.set_xticks(positions, [("0" if r["lower"] == 0 else f"2^{p}")
                                  for p, r in zip(positions, d["bins"])])
        ax.set_xlabel(f"{x} (bin lower edge)")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
        ax.set_xlabel(f"{x}: log2 bin; orange is zero")


def year_ticks(ax, years):
    years = sorted(set(years))
    if len(years) <= 5:
        ax.set_xticks(years)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("UTC year")


def save(fig, folder, name, title, scope):
    fig.suptitle(title, x=.025, ha="left", fontsize=15, fontweight="bold")
    fig.text(.025, .018, scope, color=GRAY, fontsize=9)
    fig.tight_layout(rect=(.015, .045, .99, .955), pad=1.6)
    fig.savefig(folder / name, dpi=125)
    plt.close(fig)


def raw_plots(folder, cat, raw, relation, product, scope):
    rev, meta = raw["reviews"], raw["metadata"]
    fig, axs = plt.subplots(3, 3, figsize=(12, 8.8))
    a = axs.ravel()
    stars = rev["numeric"]["rating"]
    m = {int(r["lower"]): r["numerator"] for r in stars["bins"]}
    counts(a[0], list("12345"), [m.get(i, 0) for i in range(1, 6)], stars["valid"], "Review rating")
    log_bins(a[1], rev["numeric"]["title_chars"], "Nonblank titles", "Code points")
    log_bins(a[2], rev["numeric"]["body_chars"], "Nonblank bodies", "Code points")
    log_bins(a[3], rev["numeric"]["helpful_vote"], "Cumulative helpful votes", "Votes")
    states = rev["verified_states"]
    counts(a[4], ["False", "True", "Unusable"], [r["numerator"] for r in states], rev["total"], "Verified purchase",
           "Share of parsed reviews")
    style(a[5], "Calendar coverage", rev["valid_time"]["numerator"], "Reviews")
    if rev["years"]:
        a[5].bar([r["year"] for r in rev["years"]], [r["numerator"] for r in rev["years"]], color=BLUE)
        year_ticks(a[5], [r["year"] for r in rev["years"]])
    else:
        empty(a[5])
    volumes = product["raw_volume_groups"]
    counts(a[6], list(volumes), [r["numerator"] for r in volumes.values()], product["total"],
           "Raw reviews per product", "Share of file products")
    log_bins(a[7], meta["numeric"]["price"], "Snapshot price", "Price")
    log_bins(a[8], meta["numeric"]["rating_number"], "Snapshot rating count", "Ratings")
    save(fig, folder, "univariate.png", f"{cat} | Univariate", scope + "  /  Parsed file records")

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.7))
    a = axs.ravel()
    by_star = relation["by_star"]
    star = [r["star"] for r in by_star]
    n = relation["valid_stars"]["numerator"]
    style(a[0], "Text availability by star", n, "Share within star", True)
    style(a[1], "Body length by star", sum(r["body_length"]["valid"] for r in by_star), "Code points (log scale)")
    style(a[2], "Votes and purchase by star", n, "Conditional share", True)
    if not n:
        for ax in a[:3]:
            empty(ax)
    else:
        for key, color, offset in (("title", BLUE, -.14), ("body", TEAL, .14)):
            a[0].scatter([x+offset for x in star], finite_points(r[key][-1]["fraction"] for r in by_star), color=color, label=key)
        a[0].legend(frameon=False, fontsize=8)
        for r in by_star:
            summary = r["body_length"]
            if summary["valid"]:
                q = summary["quantile_bounds"]
                a[1].vlines(r["star"], q["p25"][0], q["p75"][1], color=GRAY, linewidth=1.2)
                a[1].vlines(r["star"], *q["p50"], color=TEAL, linewidth=4, alpha=.85)
        if any(r["body_length"]["valid"] for r in by_star):
            a[1].set_yscale("log")
        else:
            empty(a[1])
        a[2].scatter([x-.14 for x in star], finite_points(r["helpful_vote"]["zero"]["fraction"] for r in by_star),
                     color=RUST, label="Zero / valid votes")
        a[2].scatter([x+.14 for x in star], finite_points(r["verified"][1]["fraction"] for r in by_star),
                     color=BLUE, label="Verified / all")
        a[2].legend(frameon=False, fontsize=8)
        for ax in a[:3]:
            ax.set_xticks(star)
            ax.set_xlabel("Stars")
    by_year = relation["by_year"]
    style(a[3], "Rating mix by year", relation["valid_rating_and_time"]["numerator"], "Share within UTC year", True)
    if by_year:
        for i, color in enumerate(PALETTE):
            a[3].scatter([r["year"] for r in by_year], [r["distribution"][i]["fraction"] for r in by_year],
                         label=str(i+1), color=color, s=14)
        a[3].legend(frameon=False, ncol=5, fontsize=7, title="Stars")
        year_ticks(a[3], [r["year"] for r in by_year])
    else:
        empty(a[3])
    rows = product["volume_vs_rating"]
    valid_products = sum(r["eligible_products"] for r in rows)
    style(a[4], "Low stars vs. product volume", valid_products, "Low-star share", True)
    if valid_products:
        for key, color, off, label in (("product_weighted_mean", TEAL, -.12, "Product-weighted"),
                                       ("review_weighted_low_star", BLUE, .12, "Review-weighted")):
            a[4].scatter([i+off for i in range(len(rows))],
                         finite_points(r[key] if isinstance(r[key], (float, type(None))) else r[key]["fraction"] for r in rows),
                         color=color, label=label)
        a[4].set_xticks(range(len(rows)), [r["volume_bin"] for r in rows])
        a[4].legend(frameon=False, fontsize=8)
    else:
        empty(a[4])
    joined = product["metadata_connection"]["review_states"]
    counts(a[5], ["None", "Conflict", "Unique", "No ID"], [r["numerator"] for r in joined.values()], rev["total"],
           "Review-to-metadata match", "Share of parsed reviews")
    save(fig, folder, "multivariate.png", f"{cat} | Multivariate", scope + "  /  No inferential tests")


def cohort_plot(folder, cat, results, policy, scope):
    main = results[policy]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.7))
    a = axs.ravel()
    stages = main["funnel"]
    style(a[0], "Global product eligibility", main["total"])
    if main["total"]:
        a[0].barh(list(range(len(stages))), [r["numerator"] for r in stages], color=BLUE)
        a[0].set_yticks(range(len(stages)), ["Usable ID", "Valid core", "5+ dedup", "15+ dedup", "Clear ties", "Eligible"])
        a[0].invert_yaxis()
        a[0].set_xlabel("Products")
        a[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        empty(a[0])
    target = main["future_low_share"]
    m = {int(round(r["lower"]*10)): r["numerator"] for r in target["bins"]}
    counts(a[1], [str(i) for i in range(11)], [m.get(i, 0) for i in range(11)], target["valid"],
           "Low stars in future ten", "Share of eligible products")
    log_bins(a[2], main["maturity_days"], "Label maturity", "Days after fifth")
    style(a[3], "Future low stars by early count", target["valid"])
    if target["valid"]:
        joint = main["early_future_joint"]
        matrix = [[joint[k*11+j]["fraction"] if joint[k*11+j]["denominator"] else float("nan")
                   for j in range(11)] for k in range(6)]
        heat = a[3].imshow(matrix, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
        a[3].set(xlabel="Low stars in next ten", ylabel="Low stars in first five")
        fig.colorbar(heat, ax=a[3], fraction=.05, format=PercentFormatter(1))
    else:
        empty(a[3])
    style(a[4], "Candidate-version sensitivity", main["total"], "Share of all products", True)
    if main["total"]:
        a[4].bar(list(results), [r["mutually_exclusive_outcomes"]["eligible"]["fraction"] for r in results.values()],
                 color=[TEAL, RUST])
    else:
        empty(a[4])
    sample = main["optional_product_sample"]
    style(a[5], "Maturity vs. volume (sample)", sample["size"], "Days after fifth")
    if sample["points"]:
        a[5].scatter([p["sequence_reviews"] for p in sample["points"]],
                     [p["maturity_days"] for p in sample["points"]], alpha=.55, color=TEAL)
        a[5].set(xscale="log", xlabel="Sequence reviews (log scale)")
    else:
        empty(a[5])
        if not sample["cap"]:
            a[5].texts[0].set_text("Optional sample disabled")
    save(fig, folder, "cohort.png", f"{cat} | 5 + 10 provisional cohort",
         scope + f"  /  Primary policy: {policy}  /  Label maturity is descriptive")


def comparison_plot(folder, comparisons, by_star, relations, scope):
    cats = list(comparisons)
    tick_names = [x.replace("_and_", " & ").replace("_", " ") for x in cats]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.6))
    axes = axs.ravel()
    metrics = [
        ("Review modalities", ("any_text", "review_image", "valid_rating", "valid_time"), "Parsed review denominator"),
        ("Snapshot modalities", ("meta_price", "meta_image"), "Parsed metadata denominator"),
        ("Low-star share", ("low_star",), "Valid-star review denominator"),
        ("Within-file / global retention", ("short_volume", "eligible"), "Products, by the two labelled populations"),
    ]
    for ax, (title, fields, desc) in zip(axes, metrics):
        style(ax, title, ylabel=desc, fraction=True)
        if not any(comparisons[c][key]["denominator"] for c in cats for key in fields):
            empty(ax)
            continue
        for i, key in enumerate(fields):
            ax.scatter([x + (i-(len(fields)-1)/2)*.15 for x in range(len(cats))],
                       finite_points(comparisons[c][key]["fraction"] for c in cats), color=PALETTE[i],
                       label=key.replace("_", " "), s=30)
        ax.set_xticks(range(len(cats)), tick_names, rotation=12, ha="right")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    ax = axes[4]
    style(ax, "Body availability by star and file", ylabel="Share within valid star", fraction=True)
    if any(r["nonblank_body"]["denominator"] for rows in by_star.values() for r in rows):
        for i, (cat, label) in enumerate(zip(cats, tick_names)):
            rows = by_star[cat]
            ax.scatter([r["star"] + (i-1.5)*.13 for r in rows],
                       finite_points(r["nonblank_body"]["fraction"] for r in rows),
                       color=PALETTE[i], s=23, label=label)
        ax.set(xticks=range(1, 6), xlabel="Stars")
        ax.legend(frameon=False, fontsize=7)
    else:
        empty(ax)
    ax = axes[5]
    style(ax, "Low stars by UTC year and file", ylabel="Share within valid year and file", fraction=True)
    if any(r["by_year"] for r in relations.values()):
        for i, cat in enumerate(cats):
            rows = relations[cat]["by_year"]
            ax.scatter([r["year"]+(i-1.5)*.12 for r in rows],
                       [r["low_star"]["fraction"] for r in rows], color=PALETTE[i], s=23)
        year_ticks(ax, [r["year"] for rel in relations.values() for r in rel["by_year"]])
    else:
        empty(ax)
    save(fig, folder, "category_comparison.png", "Four-category descriptive comparison", scope)


def extended_univariate_plot(folder, category, rev, meta, scope):
    """E14-E30 additional questions with strict denominators and bounded axes."""
    import textwrap
    fig,axs=plt.subplots(3,3,figsize=(14,10.5))
    a=axs.ravel()
    b=rev['rating_balance']
    counts(a[0],list('12345'),[b['all_five_classes'][str(i)]['numerator'] for i in range(1,6)],
           rev['numeric']['rating']['valid'],'Five rating classes (raw reviews)')
    log_bins(a[1],rev['numeric']['body_chars_all_strings'],'Review body lengths: all strings','Unicode code points')
    log_bins(a[2],rev['numeric']['body_words_all_strings'],'Review body word counts: all strings','Whitespace words')
    log_bins(a[3],rev['reviewer_activity']['activity_distribution'],'Reviews per observed user','Reviews per reviewer')
    avg=meta['numeric']['average_rating']
    style(a[4],'Metadata displayed average rating',avg['valid'],'Share of valid metadata ratings',True)
    if avg['valid']:
        a[4].bar([r['lower'] for r in avg['bins']],
                 [r['fraction'] for r in avg['bins']],width=.21,color=BLUE)
        a[4].set(xlabel='Stars (0.25-wide bins, 5.0 is its own final bin)',xlim=(.85,5.2))
    else: empty(a[4])
    log_bins(a[5],meta['numeric']['features_entries'],'Product feature entries','List entries')
    months=rev['months']
    style(a[6],'UTC monthly review counts',rev['valid_time']['numerator'],'Observed reviews')
    if months:
        a[6].plot(range(len(months)),[x['numerator'] for x in months],lw=1.5,marker='o',markersize=3,color=BLUE)
        tick=list(range(0,len(months),max(1,len(months)//6)))
        if len(months)-1 not in tick: tick.append(len(months)-1)
        a[6].set_xticks(tick,[months[i]['month'] for i in tick],rotation=40,ha='right')
    else: empty(a[6])
    for i,field in ((7,'main_category'),(8,'store')):
        d=meta['categorical'][field]
        rows=d['top'][:6]
        style(a[i],f'Metadata {field}: top labels',d['valid'],'Share of nonempty labels',True)
        if d['valid']:
            labels=[textwrap.shorten(x['label'].replace('\n',' '),width=18,placeholder='…') for x in rows]
            a[i].barh(labels[::-1],[x['fraction'] for x in rows][::-1],color=TEAL)
            a[i].set_xlabel('Other and rare levels omitted from view; counted in CSV')
        else: empty(a[i])
    save(fig,folder,'univariate_extended.png',f'{category} | Additional univariate EDA',
         scope+' / exact counts; log2 quantiles are intervals; raw records')


def multivariate_extended_plot(folder, category, rel, prod, scope):
    """E37-E39 evidence: transparent denominators; all pair stats, sampled scatter."""
    fig,axs=plt.subplots(2,2,figsize=(11.5,8))
    a=axs.flat
    rows=rel['year_modality_availability']['by_year']
    style(a[0],'Nonblank review body by UTC year',ylabel='Share among valid-time reviews',fraction=True)
    if rows:
        a[0].plot([r['year'] for r in rows], [r['body_nonblank']['fraction'] for r in rows],
                  marker='o',ms=3,color=TEAL)
        year_ticks(a[0],[r['year'] for r in rows])
    else: empty(a[0])
    facets=prod['metadata_conditioned_ratings']['facets']['valid_price']
    style(a[1],'Low-star reviews by snapshot price availability',ylabel='Share of valid-star reviews',fraction=True)
    labels=['Price available','Price missing']
    values=[facets[key]['low_star_review_weighted']['fraction'] for key in ('available','unavailable')]
    if any(x is not None for x in values):
        for i,x in enumerate(values):
            n=facets[('available','unavailable')[i]]['low_star_review_weighted']['denominator']
            if x is None:
                a[1].text(i,.05,'N/A (n=0)',ha='center',fontsize=8)
            else:
                a[1].bar(i,x,color=(TEAL,RUST)[i],width=.65)
                a[1].text(i,min(1.,x+.025),f'n={n:,}',ha='center',fontsize=8)
        a[1].set_xticks(range(2),labels)
    else:empty(a[1])
    pair=rel['body_length_vs_helpful_vote']
    sample=pair['display_sample']
    style(a[2],f'Body length vs helpful votes | display sample {sample["sample_size"]}',
          ylabel='log(1 + helpful votes)')
    if sample['points']:
        a[2].scatter([p['body_log1p'] for p in sample['points']],
                     [p['votes_log1p'] for p in sample['points']],s=6,alpha=.35,color=BLUE)
        a[2].set_xlabel('log(1 + nonblank body code points)')
    else:empty(a[2])
    style(a[3],'Within-star Pearson on paired log values',ylabel='Pearson r')
    rows=pair['by_star'];valid=[r for r in rows if r['correlation'] is not None]
    if valid:
        a[3].bar([r['star'] for r in valid],[r['correlation'] for r in valid],color=TEAL)
        a[3].set(ylim=(-1.05,1.05),xticks=range(1,6),xlabel='Stars (only defined correlations displayed)')
        a[3].axhline(0,color=GRAY,lw=.8)
    else:empty(a[3])
    save(fig,folder,'multivariate_extended.png',f'{category} | Conditional relationships',
         scope+' / snapshot metadata; observational associations; sampled scatter, full-pair Pearson')


def missingness_by_subgroup_plot(folder, category, relations, scope):
    """E37 additional plot: raw year/star subgroups, with checked denominators."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    year_rows = relations['year_modality_availability']['by_year']
    style(axs[0], 'Review text/image availability by UTC year',
          ylabel='Share among checked reviews with valid timestamp', fraction=True)
    if year_rows:
        for key, color, label in [('body_nonblank',TEAL,'Nonblank body'),
                                  ('review_image_url',BLUE,'Syntactically valid image URL')]:
            x=[r['year'] for r in year_rows]
            y=[r[key]['fraction'] if r[key]['fraction'] is not None else float('nan') for r in year_rows]
            axs[0].plot(x,y,marker='o',ms=3,color=color,label=label)
        year_ticks(axs[0],[r['year'] for r in year_rows])
        axs[0].legend(fontsize=8,frameon=False)
    else:
        empty(axs[0])
    by_star=relations['by_star']
    style(axs[1], 'Review text/image availability by valid star',
          ylabel='Share among reviews with valid star (checked)', fraction=True)
    if any(r['valid_rating_records'] for r in by_star):
        xs=[r['star'] for r in by_star]
        body=[r['body'][-1]['fraction'] for r in by_star]
        images=[r['review_image_url']['fraction'] for r in by_star]
        axs[1].plot(xs,[x if x is not None else float('nan') for x in body],
                    marker='o',color=TEAL,label='Nonblank body')
        axs[1].plot(xs,[x if x is not None else float('nan') for x in images],
                    marker='s',color=BLUE,label='Syntactically valid image URL')
        axs[1].set_xticks(range(1,6));axs[1].set_xlabel('Valid stars')
        axs[1].legend(fontsize=8,frameon=False)
    else:
        empty(axs[1])
    save(fig,folder,'multivariate_missingness.png',
         f'{category} | Missingness by year and rating (E37)',
         scope+' / raw records; missingness mechanism not inferred')
