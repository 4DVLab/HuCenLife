import re


class ActionLevel:
    def __init__(self,level_file_path):
        with open(level_file_path,"r",encoding='utf-8') as f:
            gtaction = f.read()
        actions = re.findall(r"LV'(\d)''(.*?)'{\n([\s\S]*?)\n}",gtaction)
        self.actions = {}
        self.sqactions = {}
        self.action_name_list = []
        counter = 0
        for action in actions:
            if action[1] not in self.action_name_list:
                self.action_name_list.append(action[1])
            for a in action[2].split("\n"):
                self.actions[a.strip()] = [int(action[0]),action[1]]
                self.sqactions[a.strip()] = [str(int(action[0])).zfill(3)+str(counter).zfill(3),action[1]]
            counter += 1
        self.unknow_action = []

        # actions_list = []
        # for key in self.actions.keys():
        #     actions_list.append([key,self.actions[key]])
        # actions_list.sort(key=lambda x:x[0])
        # self.actions_leveled = []
        # for action in actions_list:
        #     self.actions_leveled.append(action[1])
        # self.levels = len(self.actions_leveled)

    def get_action_level(self,action):
        if action in self.actions.keys():
            return self.actions[action][0],self.actions[action][1]
        else:
            if action not in self.unknow_action:
                print("UNKOWN ACTION: ", action)
                self.unknow_action.append(action)
            return (-1,"ACTION_CONFLICT")

    def get_action_level_sq(self,action):
        if action in self.sqactions.keys():
            return self.sqactions[action][0],self.sqactions[action][1]
        else:
            if action not in self.unknow_action:
                print("UNKOWN ACTION: ", action)
                self.unknow_action.append(action)
            return (-1,"ACTION_CONFLICT")

    def get_action(self,action_list):
        action_level = []
        for action in action_list:
            action_level.append(self.get_action_level(action.strip()))
        action_level.sort(key=lambda x:x[0])
        for i in range(1,len(action_level)):
            if action_level[0][0] == action_level[i][0]:
                if action_level[0][1] != action_level[i][1]:
                    return "ACTION_CONFLICT"
                    # return action_level[0][1]
            else:
                # if action_level[0][1] == "OTHER":
                #     print(action_list)
                return action_level[0][1]
        # if action_level[0][1]=="OTHER":
        #     print(action_list)
        return action_level[0][1]

    def get_action_alpha(self,action_list):
        action_level = []
        for action in action_list:
            action_level.append(self.get_action_level_sq(action.strip()))
        action_level.sort(key=lambda x:x[0])
        for i in range(1,len(action_level)):
            if action_level[0][0] == action_level[i][0]:
                if action_level[0][1] != action_level[i][1]:
                    # return "ACTION_CONFLICT"
                    return action_level[0][1]
            else:
                return action_level[0][1]
        return action_level[0][1]

    def get_problem_action(self,action_list):
        action_level = []
        for action in action_list:
            action_level.append(self.get_action_level(action.strip()))
        action_level.sort(key=lambda x:x[0])
        for i in range(1,len(action_level)):
            if action_level[0][0] == action_level[i][0]:
                if action_level[0][1] != action_level[i][1]:
                    return "ACTION_CONFLICT"
                    # return action_level[0][1]
            else:
                return action_level[0][1]
        return action_level[0][1]

    def get_problem_level_action(self,action_list):
        action_level = []
        for action in action_list:
            action_level.append(self.get_action_level(action))
        action_level.sort(key=lambda x:x[0])
        problem_level_action = [action_level[0][1]]
        for i in range(1,len(action_level)):
            if action_level[0][0] == action_level[i][0]:
                if action_level[i][1] not in problem_level_action:
                    problem_level_action.append(action_level[i][1])
        problem_level_action.sort()
        return str(problem_level_action)
    def action_length(self):
        action_set = []
        for key in self.sqactions.keys():
            action_set.append(self.sqactions[key][1])
        return len(set(action_set))
    def action_list(self):
        return self.action_name_list
if __name__ == "__main__":
    action_level = ActionLevel(r"actionlist_merge.txt")
    print(action_level.get_action_level('pull the car'))
    print(action_level.get_action(['pull the car']))
    print(action_level.get_action(['pull the car','pull the child']))
    print(action_level.get_action(['pull the car','test']))
    print(action_level.get_action(['test',"pull something",'pull the car','pull the child']))
    print(action_level.get_action(['train','pull the car', 'test']))
    print(action_level.action_length())