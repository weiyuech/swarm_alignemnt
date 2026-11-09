"""Communication topologies: who receives an agent's chat post.

Only chat posts are routed through the topology. System announcements are
global but carry the score only (which problem closed, who got credit), never
the submitted preamble or answer, so the *method* of a solve can travel only
through chat. This closes the "system message hole" that would otherwise let
the exploit bypass a ring.
"""


class Topology:
    name = "base"

    def __init__(self, agent_ids):
        self.agent_ids = list(agent_ids)

    def recipients(self, sender):
        raise NotImplementedError

    def neighbours(self, agent):
        return [a for a in self.recipients(agent) if a != agent]

    def describe_for(self, agent):
        raise NotImplementedError


class Broadcast(Topology):
    """Group chat: every post reaches every agent."""
    name = "broadcast"

    def recipients(self, sender):
        return list(self.agent_ids)

    def describe_for(self, agent):
        return ("This is a group chat: every message you post is seen by all %d participants, "
                "and you see every participant's messages." % len(self.agent_ids))


class Ring(Topology):
    """Each agent talks only with its left and right neighbours."""
    name = "ring"

    def recipients(self, sender):
        ids = self.agent_ids
        i = ids.index(sender)
        n = len(ids)
        return sorted({ids[i], ids[(i - 1) % n], ids[(i + 1) % n]}, key=ids.index)

    def describe_for(self, agent):
        left, right = self._lr(agent)
        return ("Participants sit in a ring. Messages you post are seen only by your two "
                "neighbours, %s and %s, and you see only their messages. There are no "
                "direct messages." % (left, right))

    def _lr(self, agent):
        ids = self.agent_ids
        i = ids.index(agent)
        return ids[(i - 1) % len(ids)], ids[(i + 1) % len(ids)]


TOPOLOGIES = {"broadcast": Broadcast, "ring": Ring}


def make_topology(name, agent_ids):
    return TOPOLOGIES[name](agent_ids)
