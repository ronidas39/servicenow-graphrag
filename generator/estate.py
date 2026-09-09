"""Generate a believable enterprise IT estate for the GraphRAG article.

The estate is the thing everything else hangs off. Incidents point at configuration
items, changes point at them too, and the dependency edges between them are what the
graph article is actually about. If the estate is a clean tree, the graph looks better
than any real one ever does, and the whole comparison is worthless.

So this deliberately builds the shapes a real CMDB has and a generated one usually does
not: a few items that everything depends on, at least one dependency loop, edges that
were last verified years ago, and a long tail of small services nobody thinks about.

Author: Roni Das
Created: 2026-09-08
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# Real class names, read from the instance rather than invented. A configuration item
# is stored in the table for its class, and it also belongs to every parent table above
# it, which is what lets a query ask for "any server" without listing every kind.
# ⛔ EVERY CLASS HERE IS INDEPENDENTLY IDENTIFIABLE, AND THAT IS NOT A DETAIL.
# ServiceNow splits configuration item classes in two. An independent class has its own
# identity, so a server can be recognised from its own attributes. A DEPENDENT class
# cannot: two applications both called "payments-api" are only different things if you
# say which host each one runs on, so the identification engine refuses a dependent item
# unless the payload also carries its container and the containment relationship.
#
# Measured on the instance, one item at a time:
#   independent : cmdb_ci_server, cmdb_ci_computer, cmdb_ci_linux_server,
#                 cmdb_ci_win_server, cmdb_ci_service, cmdb_ci_cluster, cmdb_ci_lb,
#                 cmdb_ci_storage_server
#   dependent   : cmdb_ci_appl, cmdb_ci_db_instance, cmdb_ci_database,
#                 cmdb_ci_app_server, cmdb_ci_web_server, cmdb_ci_storage_device
#                 -> "In payload no relations defined for dependent class [X]"
#
# The first version of this estate used cmdb_ci_appl and cmdb_ci_db_instance, and every
# batch was rejected. The application layer is modelled as an application service, which
# is what the modern service model calls it anyway, and databases as their own servers.
CLASSES: dict[str, str] = {
    "linux": "cmdb_ci_linux_server",
    "windows": "cmdb_ci_win_server",
    "app": "cmdb_ci_service",
    "db": "cmdb_ci_server",
    "service": "cmdb_ci_service",
    "loadbalancer": "cmdb_ci_lb",
    "cluster": "cmdb_ci_cluster",
    "storage": "cmdb_ci_storage_server",
}

# Relationship type names are stored as "parent descriptor::child descriptor". Reading
# them backwards inverts every dependency answer, so the generator writes them the same
# way ServiceNow does and the loader keys on the type record, never on this string.
REL_DEPENDS = "Depends on::Used by"
REL_RUNS_ON = "Runs on::Runs"
REL_HOSTED = "Hosted on::Hosts"
REL_IP = "IP Connection::IP Connection"

# ⛔ RELATIONSHIPS THAT DO NOT CARRY IMPACT, AND THE ESTATE NEEDS THEM.
# A rack contains a server. The server does not fail because the rack was moved. A team
# manages a service. The service does not fail because the team reorganised.
# A first version of this estate had only the three types above, so 17,969 of 17,973
# edges propagated impact and the filter the article argues for excluded four edges out
# of eighteen thousand. The lesson was invisible because the data could not show it.
REL_RACK = "In Rack::Rack contains"
REL_MANAGED = "Managed by::Manages"
# ⛔ WRITTEN BACKWARDS, IN AN ARTICLE ABOUT RELATIONSHIP DIRECTION. The instance
# has "Owns::Owned by": the PARENT owns, the CHILD is owned. I inverted it from
# memory and the loader refused the whole phase, which is the correct outcome and
# the reason it resolves every type by name before writing a single row.
REL_OWNED = "Owns::Owned by"
REL_LOCATED = "Located in Zone::Zone contains"

ENVIRONMENTS = ["prd", "stg", "dev", "dr"]
REGIONS = ["eu-west", "us-east", "ap-south"]

# A real estate is mostly production, because production is what gets built out and
# monitored. Development is numerous but small.
ENV_WEIGHTS = [0.42, 0.18, 0.28, 0.12]

BUSINESS_DOMAINS = [
    "payments", "identity", "checkout", "inventory", "pricing",
    "loyalty", "search", "notifications", "reporting", "fraud",
    "settlement", "onboarding", "billing", "catalogue", "shipping",
]


@dataclass
class CI:
    """One configuration item, in the shape ServiceNow stores it."""

    key: str
    name: str
    sys_class_name: str
    environment: str
    region: str
    domain: str
    operational_status: int = 1
    install_status: int = 1
    # A real CMDB records where a record came from and when it was last confirmed.
    # Both fields matter later: the article shows an answer that cites its own staleness.
    discovery_source: str = "Discovery"
    last_discovered: datetime | None = None
    sys_id: str | None = None


@dataclass
class Rel:
    """One dependency edge. Parent and child are not interchangeable."""

    parent_key: str
    child_key: str
    type_name: str
    last_discovered: datetime | None = None


@dataclass
class Estate:
    cis: list[CI] = field(default_factory=list)
    rels: list[Rel] = field(default_factory=list)

    def by_class(self, kind: str) -> list[CI]:
        want = CLASSES[kind]
        return [c for c in self.cis if c.sys_class_name == want]


# ⛔ A DATASET IS A SNAPSHOT AS OF A MOMENT, AND THE MOMENT MUST BE FIXED. Timestamps
# were derived from datetime.now(), so two builds from the same seed produced different
# files and different checksums, while the manifest published those checksums and the
# article calls the dataset reproducible. It was not. Anchoring to a stated date is also
# what a real export does: you export "as of" a time, and everything downstream measures
# staleness against that, not against whenever somebody happened to run the script.
AS_OF = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)


def _now() -> datetime:
    return AS_OF


def build_estate(
    services: int = 220,
    seed: int = 20260908,
    stale_fraction: float = 0.18,
) -> Estate:
    """Build the estate.

    Args:
        services: how many business services to build the estate around. Everything
            else scales from this, because a real estate is sized by what it serves.
        seed: fixed so the dataset is reproducible.
        stale_fraction: share of dependency edges last confirmed over a year ago. Real
            CMDB relationship data rots, and the article measures what that costs.

    Returns:
        An Estate with configuration items and the dependencies between them.
    """
    rng = random.Random(seed)
    est = Estate()
    now = _now()

    def pick_env() -> str:
        return rng.choices(ENVIRONMENTS, weights=ENV_WEIGHTS, k=1)[0]

    def stamp() -> datetime:
        # Most items were seen recently. A meaningful minority were not.
        if rng.random() < stale_fraction:
            return now - timedelta(days=rng.randint(400, 1500))
        return now - timedelta(days=rng.randint(0, 45))

    def add(key: str, name: str, kind: str, env: str, region: str, domain: str) -> CI:
        ci = CI(
            key=key,
            name=name,
            sys_class_name=CLASSES[kind],
            environment=env,
            region=region,
            domain=domain,
            last_discovered=stamp(),
            # ⛔ THESE ARE CHOICE VALUES, NOT FREE TEXT. Read off sys_choice for
            # cmdb_ci.discovery_source on the live instance. Inventing one makes the
            # identification engine reject the whole payload with INVALID_INPUT_DATA.
            discovery_source=rng.choice(
                ["ServiceNow", "ServiceNow", "ServiceNow", "ServiceWatch", "Manual Entry"]
            ),
        )
        est.cis.append(ci)
        return ci

    # Shared infrastructure comes first, because everything else attaches to it. These
    # become the items with thousands of edges, which is the shape that breaks an
    # unbounded traversal and is the reason the article caps hop count.
    clusters: list[CI] = []
    for region in REGIONS:
        for n in range(1, 3):
            key = f"cluster-{region}-{n:02d}"
            clusters.append(add(key, key, "cluster", "prd", region, "platform"))
    storage: list[CI] = []
    for region in REGIONS:
        key = f"san-{region}-01"
        storage.append(add(key, key, "storage", "prd", region, "platform"))

    # ⛔ THE NAMES MUST NOT SPELL OUT THE GRAPH. An audit measured a plain substring
    # search recovering a whole service stack at 78% recall and 78% precision, with no
    # graph involved at all. The cause was one line: every item in a stack was named
    # from the same "{domain}-{env}-{index}" token, so finding the service found its
    # application, its hosts and its database too. The question the article says a graph
    # is needed for was answerable by string matching, because this file had written the
    # answer into the text.
    #
    # Real estates do not look like that. A service carries a business name because
    # people talk about it. A host carries an infrastructure name because nobody does.
    # Numbering is per kind of thing, not per service, so no token runs down a stack.
    # Getting from a service to its hosts now requires following an edge.
    seq: dict[str, int] = {}

    def infra_name(prefix: str) -> str:
        seq[prefix] = seq.get(prefix, 0) + 1
        return f"{prefix}{seq[prefix]:04d}"

    for idx in range(services):
        domain = rng.choice(BUSINESS_DOMAINS)
        env = pick_env()
        region = rng.choice(REGIONS)
        tag = f"{domain}-{env}-{idx:03d}"

        # ⛔ THE NAME MUST BE UNIQUE, NOT JUST THE KEY. A first version named these
        # "payments service (prd)" without the number, so every service in the same
        # domain and environment shared one name: 120 names were reused, one of them 61
        # times. The identification engine identifies on name, so a batch containing two
        # of them is abandoned, and a reader looking at the graph sees 61 identical
        # labels. Unique keys did not save it, because nothing identifies on the key.
        svc = add(f"svc-{tag}", f"{domain} service {idx:03d} ({env})",
                  "service", env, region, domain)
        app = add(f"app-{tag}", infra_name("app"), "app", env, region, domain)
        est.rels.append(Rel(svc.key, app.key, REL_DEPENDS, stamp()))

        # Applications run on servers. How many depends on the environment, because
        # production is built for failure and development is not.
        host_count = {"prd": rng.randint(2, 5), "dr": 2, "stg": 2, "dev": 1}[env]
        hosts: list[CI] = []
        for h in range(host_count):
            kind = "linux" if rng.random() < 0.82 else "windows"
            host = add(f"host-{tag}-{h}",
                       infra_name("lnx" if kind == "linux" else "win"),
                       kind, env, region, domain)
            hosts.append(host)
            est.rels.append(Rel(app.key, host.key, REL_RUNS_ON, stamp()))
            # Every host sits on shared kit. This is where the shared items pick up
            # their very high edge count.
            # ⛔ THE HOSTED THING IS THE PARENT. A ServiceNow relationship type is named
            # "what the parent is to the child::what the child is to the parent", so
            # "Hosted on::Hosts" means the PARENT is hosted on the CHILD. Writing the
            # container as parent says the cluster is hosted on the server, which is the
            # sentence backwards, and it cost the graph its whole upward direction: a
            # cluster ended up with 950 dependencies and nothing depending on it, so
            # "what breaks if this cluster fails" correctly returned nothing.
            est.rels.append(
                Rel(host.key,
                    rng.choice([c for c in clusters if c.region == region]).key,
                    REL_HOSTED, stamp())
            )

        # Most services own a database. Some share one, which is how a single database
        # ends up on the blast radius of several unrelated services.
        if rng.random() < 0.72:
            db = add(f"db-{tag}", infra_name("pg"), "db", env, region, domain)
            est.rels.append(Rel(app.key, db.key, REL_DEPENDS, stamp()))
            # Same correction: the database is hosted on the storage, not the reverse.
            est.rels.append(
                Rel(db.key,
                    rng.choice([s for s in storage if s.region == region]).key,
                    REL_HOSTED, stamp())
            )

        if env == "prd" and rng.random() < 0.6:
            lb = add(f"lb-{tag}", infra_name("lb"), "loadbalancer", env, region, domain)
            est.rels.append(Rel(lb.key, app.key, REL_DEPENDS, stamp()))

    # Services call each other. This is what makes a question multi step, and it is the
    # part a table cannot answer.
    svcs = est.by_class("service")
    for svc in svcs:
        for _ in range(rng.choices([0, 1, 2, 3], weights=[0.30, 0.38, 0.22, 0.10])[0]):
            other = rng.choice(svcs)
            if other.key != svc.key and other.environment == svc.environment:
                est.rels.append(Rel(svc.key, other.key, REL_DEPENDS, stamp()))

    # Racks and zones. These are real relationships that a CMDB records and that carry
    # no impact at all, which is exactly why they are here.
    racks = [add(f"rack-{r}-{n:02d}", f"rack-{r}-{n:02d}", "cluster", "prd", r, "platform")
             for r in REGIONS for n in range(1, 5)]
    servers = [c for c in est.cis
               if c.sys_class_name in (CLASSES["linux"], CLASSES["windows"])]
    for srv in servers:
        rack = rng.choice([x for x in racks if x.region == srv.region])
        # "In Rack::Rack contains" and "Located in Zone::Zone contains" name the parent
        # first as well, so the server is the parent of both. The rack standing in for a
        # zone is a simplification this estate makes on purpose, and Part 4 says so.
        est.rels.append(Rel(srv.key, rack.key, REL_RACK, stamp()))
        if rng.random() < 0.30:
            est.rels.append(Rel(srv.key, rack.key, REL_LOCATED, stamp()))

    # Ownership and management. Also real, also carrying no impact.
    for svc in est.by_class("service"):
        # "Managed by::Manages" puts the managed thing first, so the service is the
        # parent. "Owns::Owned by" puts the OWNER first, so this one was already the
        # right way round and is deliberately left alone: the fix is per type name, not
        # a rule that every edge was backwards.
        if rng.random() < 0.55:
            est.rels.append(Rel(svc.key, rng.choice(racks).key, REL_MANAGED, stamp()))
        if rng.random() < 0.35:
            est.rels.append(Rel(rng.choice(racks).key, svc.key, REL_OWNED, stamp()))

    # At least one genuine loop. Real estates have them, usually through a shared
    # authentication or logging service, and a naive traversal never comes back.
    ring = [s for s in svcs if s.environment == "prd"][:4]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        est.rels.append(Rel(a.key, b.key, REL_IP, stamp()))

    # A few items nobody owns and nothing points at. Every real CMDB has these.
    for n in range(max(4, services // 30)):
        add(f"orphan-{n:03d}", f"unclaimed-host-{n:03d}", "linux",
            pick_env(), rng.choice(REGIONS), rng.choice(BUSINESS_DOMAINS))

    return est


def describe(est: Estate) -> str:
    """A short report, so the shape can be checked before anything is loaded."""
    from collections import Counter

    cls = Counter(c.sys_class_name for c in est.cis)
    rel = Counter(r.type_name for r in est.rels)
    degree: Counter[str] = Counter()
    for r in est.rels:
        degree[r.parent_key] += 1
        degree[r.child_key] += 1

    now = _now()
    stale = sum(
        1 for r in est.rels
        if r.last_discovered and (now - r.last_discovered).days > 365
    )

    lines = [
        f"configuration items : {len(est.cis):,}",
        f"dependency edges    : {len(est.rels):,}",
        "",
        "by class:",
        *[f"  {k:28s} {v:>6,}" for k, v in cls.most_common()],
        "",
        "by relationship type:",
        *[f"  {k:28s} {v:>6,}" for k, v in rel.most_common()],
        "",
        f"stale edges (over a year old) : {stale:,} ({stale / max(1, len(est.rels)):.0%})",
        "busiest items (these are the ones that break a traversal):",
        *[f"  {k:28s} {v:>6,} edges" for k, v in degree.most_common(5)],
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(build_estate()))
