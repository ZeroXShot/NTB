"""What one module looks like in space, and what its geometry connects.

The studio draws the *authored* document, but a generator's instances are not
authored nodes and a rule's edges are not authored edges. Rather than have the
frontend reimplement where an instance sits, the server answers that here, from
the same code that lowers the document.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ntb.ir.document import Document, Module
from ntb.ir.graph import Generator, Node
from ntb.spatial.resolve import MODULE_OP
from ntb.spatial.rules import Placed, RuleError, derive_pairs


class BlockKind(StrEnum):
    #: A node the author placed.
    NODE = "node"
    #: A node that instantiates another module.
    INSTANCE = "instance"
    #: One repetition produced by a generator. Nobody placed it.
    GENERATED = "generated"


class LinkKind(StrEnum):
    #: An edge the author drew.
    EDGE = "edge"
    #: An edge a spatial rule derived from where the blocks sit.
    RULE = "rule"
    #: The wiring a generator adds between consecutive instances.
    CHAIN = "chain"


@dataclass(frozen=True, slots=True)
class Block:
    key: str
    label: str
    op: str
    kind: BlockKind
    pos: tuple[float, float, float]
    extent: tuple[float, float, float]
    source: str = ""
    index: int | None = None


@dataclass(frozen=True, slots=True)
class Link:
    src: str
    dst: str
    kind: LinkKind
    source: str = ""


@dataclass(frozen=True, slots=True)
class Preview:
    blocks: tuple[Block, ...] = ()
    links: tuple[Link, ...] = ()
    #: Rules that could not be applied. Reported, never raised: a half-placed
    #: document is the normal state of one being edited.
    problems: tuple[str, ...] = ()


def preview(document: Document, module_id: str | None = None) -> Preview:
    """Blocks and links for one module, generated ones included."""
    module = document.module(module_id or document.root)
    if module is None:
        return Preview()

    blocks: list[Block] = [_node_block(node) for node in module.nodes]
    links: list[Link] = [
        Link(src=edge.src.node, dst=edge.dst.node, kind=LinkKind.EDGE, source=edge.id)
        for edge in module.edges
    ]

    for generator in module.generators:
        instances = _instances(document, generator)
        blocks.extend(instances)
        if generator.chain:
            links.extend(
                Link(
                    src=instances[i].key,
                    dst=instances[i + 1].key,
                    kind=LinkKind.CHAIN,
                    source=generator.id,
                )
                for i in range(len(instances) - 1)
            )

    by_key = {block.key: block for block in blocks}
    problems: list[str] = []
    for rule in module.spatial_rules:
        members = _members(rule.members, blocks)
        if len(members) < 2:
            problems.append(f"rule {rule.id!r} covers fewer than two blocks")
            continue
        try:
            pairs = derive_pairs(rule, [Placed(key=b.key, pos=b.pos) for b in members])
        except RuleError as exc:
            problems.append(str(exc))
            continue
        links.extend(
            Link(
                src=members[source].key,
                dst=members[target].key,
                kind=LinkKind.RULE,
                source=rule.id,
            )
            for source, target in pairs
        )

    return Preview(
        blocks=tuple(blocks),
        links=tuple(link for link in links if link.src in by_key and link.dst in by_key),
        problems=tuple(problems),
    )


def _node_block(node: Node) -> Block:
    kind = BlockKind.INSTANCE if node.op == MODULE_OP else BlockKind.NODE
    return Block(
        key=node.id,
        label=node.name or node.id,
        op=node.op,
        kind=kind,
        pos=node.placement.pos,
        extent=node.placement.extent,
        source=str(node.attrs.get("module", "")) if kind is BlockKind.INSTANCE else "",
    )


def _instances(document: Document, generator: Generator) -> list[Block]:
    child = document.module(generator.module)
    extent = _module_extent(child)
    offset = generator.axis.offset
    blocks: list[Block] = []
    for index in range(generator.count):
        pos = list(generator.origin)
        pos[offset] += index * generator.step
        blocks.append(
            Block(
                key=f"{generator.id}-{index}",
                label=f"{generator.id}[{index}]",
                op=generator.module,
                kind=BlockKind.GENERATED,
                pos=(pos[0], pos[1], pos[2]),
                extent=extent,
                source=generator.id,
                index=index,
            )
        )
    return blocks


def _module_extent(module: Module | None) -> tuple[float, float, float]:
    """A generated block is drawn as one box, sized to what it contains."""
    if module is None or not module.nodes:
        return (1.0, 1.0, 1.0)
    spans = []
    for axis in range(3):
        low = min(n.placement.pos[axis] - n.placement.extent[axis] / 2 for n in module.nodes)
        high = max(n.placement.pos[axis] + n.placement.extent[axis] / 2 for n in module.nodes)
        spans.append(max(high - low, 0.5))
    return (spans[0], spans[1], spans[2])


def _members(names: tuple[str, ...], blocks: list[Block]) -> list[Block]:
    """Resolve rule members, where a generator id stands for its instances."""
    by_key = {block.key: block for block in blocks}
    resolved: list[Block] = []
    for name in names:
        if name in by_key:
            resolved.append(by_key[name])
            continue
        resolved.extend(b for b in blocks if b.source == name and b.kind is BlockKind.GENERATED)
    return resolved
