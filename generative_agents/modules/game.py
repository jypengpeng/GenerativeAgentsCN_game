"""generative_agents.game"""

import os
import copy
import threading

from modules.utils import GenerativeAgentsMap, GenerativeAgentsKey
from modules import utils
from .maze import Maze
from .agent import Agent


class Game:
    """The Game"""

    def __init__(self, name, static_root, config, conversation, logger=None):
        self.name = name
        self.static_root = static_root
        self.record_iterval = config.get("record_iterval", 30)
        self.logger = logger or utils.IOLogger()
        self.maze = Maze(self.load_static(config["maze"]["path"]), self.logger)
        self.conversation = conversation
        self.conversation_lock = threading.Lock()
        self.agents = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        storage_root = os.path.join(f"results/checkpoints/{name}", "storage")
        if not os.path.isdir(storage_root):
            os.makedirs(storage_root)
        for name, agent in config["agents"].items():
            agent_config = utils.update_dict(
                copy.deepcopy(agent_base), self.load_static(agent["config_path"])
            )
            agent_config = utils.update_dict(agent_config, agent)

            agent_config["storage_root"] = os.path.join(storage_root, name)
            self.agents[name] = Agent(agent_config, self.maze, self.conversation, self.logger)

    def get_agent(self, name):
        return self.agents[name]

    def get_presences(self):
        presences = []
        for agent in self.agents.values():
            presences.append(agent.get_presence())
        return presences

    def _rule_group(self, presences):
        # 规则分组：先按 arena，再按目标/邻近合并
        by_arena = {}
        for p in presences:
            key = p.get("arena") or "<none>"
            by_arena.setdefault(key, []).append(p)

        groups = []
        for _, lst in by_arena.items():
            # 简化：同一 arena 作为一组；如需更细可再按 target/coord 划分
            groups.append([p["name"] for p in lst])
        return groups

    def group_agents(self, use_llm=False):
        presences = self.get_presences()
        # 预留：可在此调用 LLM 分组并回退规则分组
        return self._rule_group(presences)

    def agent_think(self, name, status):
        agent = self.get_agent(name)
        plan = agent.think(status, self.agents)
        info = {
            "currently": agent.scratch.currently,
            "associate": agent.associate.abstract(),
            "concepts": {c.node_id: c.abstract() for c in agent.concepts},
            "chats": [
                {"name": "self" if n == agent.name else n, "chat": c}
                for n, c in agent.chats
            ],
            "action": agent.action.abstract(),
            "schedule": agent.schedule.abstract(),
            "address": agent.get_tile().get_address(as_list=False),
            "presence": agent.status.get("presence", agent.get_presence()),
        }
        if (
            utils.get_timer().daily_duration() - agent.last_record
        ) > self.record_iterval:
            info["record"] = True
            agent.last_record = utils.get_timer().daily_duration()
        else:
            info["record"] = False
        if agent.llm_available():
            info["llm"] = agent._llm.get_summary()
        title = "{}.summary @ {}".format(
            name, utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        )
        self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))
        return {"plan": plan, "info": info}

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))

    def reset_game(self):
        # 并行初始化各 Agent（主要是创建 LLM 句柄等）
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(len(self.agents), (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(agent.reset): name for name, agent in self.agents.items()}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        self.logger.warning("agent reset failed for {}: {}".format(futures[fut], e))
        except Exception as e:
            self.logger.warning("parallel reset encountered error: {}".format(e))

        # 顺序记录初始化状态日志
        for a_name, agent in self.agents.items():
            title = "{}.reset".format(a_name)
            self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))


def create_game(name, static_root, config, conversation, logger=None):
    """Create the game"""

    utils.set_timer(**config.get("time", {}))
    GenerativeAgentsMap.set(GenerativeAgentsKey.GAME, Game(name, static_root, config, conversation, logger=logger))
    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)


def get_game():
    """Get the gloabl game"""

    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)
