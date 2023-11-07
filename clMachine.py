# =========================================================================================
class Machine():
    """Machine in factory - properties"""

    def __init__(self, aName,time):
        self.name = aName
        self.currentTime = time    
        self.assignedOpera = []
        self.running = 1

    def exportToDict(self):
        """Serialize information about Machine into dictionary"""
        exData = {}
        exData['machineName'] = self.name
        exData['currentTime'] = self.currentTime
        exData['assignedOper'] = self.assignedOpera
        return exData

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name
