#!/usr/bin/python

import os
import sys
import codecs
import re
import locale
sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout) 

class node() :
    def __init__(self,lex,tokenType,feats,fsList) :
        self.lex = lex
        self.tokenType = tokenType
        self.__attributes = {}
        self.errors = []
        self.name = None
        self.updateAttributes(feats)
        self.parent = None
        self.parentRelation = None
        self.alignedTo = None
        self.fsList = fsList
        
    def updateAttributes(self,feats) :
        for attribute in feats.keys() :
            self.__attributes[attribute] = feats[attribute]
        self.assignName()

    def assignName(self) :
        if self.__attributes.has_key('name') : 
            self.name = self.getAttribute('name')
        else :
            self.errors.append('No name for this token Node')
            
    def printValue(self) :
        return self.lex

    def printSSFValue(self, prefix, allFeat) :
        returnValue = [prefix , self.printValue() , self.tokenType]
        if allFeat == False : 
            fs = ['<fs']
            for key in self.__attributes.keys() :
                fs.append(key + "='" + self.getAttribute(key) + "'")
            delim = ' '
            fs[-1] = fs[-1] + '>'
            
        else :
            fs = self.fsList
            delim = '|'
        return ('\t'.join(x for x in returnValue) + '\t' + delim.join(x for x in fs))

    def getAttribute(self,key) :
        if self.__attributes.has_key(key) :
            return self.__attributes[key]
        else :
            return None

    def addAttribute(self,key,value) :
        self.__attributes[key] = value

    def deleteAttribute(self,key) :
        del self.__attributes[key]
            
class chunkNode() :
    def __init__(self,chunkType,feats,fsList) :
        self.nodeList = []
        self.parent = '0'
        self.__attributes = {}
        self.parentRelation = 'root'
        self.name = None
        self.type = None
        self.head = None
        self.isParent = False
        self.errors = []
        self.updateAttributes(feats)
        self.updateDrel()
        self.type = chunkType
        self.fsList = fsList
        
    def updateAttributes(self,feats) :
        for attribute in feats.keys() :
            self.__attributes[attribute] = feats[attribute]
        self.assignName()

    def assignName(self) :
        if self.__attributes.has_key('name') : 
            self.name = self.getAttribute('name')
        else :
            self.errors.append('No name for this chunk Node')
        
    def updateNodeList(self,nodeList) :
        self.nodeList.extend(nodeList)

    def updateDrel(self) :
        if self.__attributes.has_key('drel') :
            drelList = self.getAttribute('drel').split(':')
            if len(drelList) == 2 :
                self.parent = drelList[1]
                self.parentRelation = self.getAttribute('drel').split(':')[0]
        elif self.__attributes.has_key('dmrel') :
            drelList = self.getAttribute('dmrel').split(':')
            if len(drelList) == 2 :
                self.parent = drelList[1]
                self.parentRelation = self.getAttribute('dmrel').split(':')[0]

    def printValue(self) :
        returnString = []
        for node in self.nodeList :
            returnString.append(node.printValue())
        return ' '.join(x for x in returnString)

    def printSSFValue(self, prefix, allFeat) :
        returnStringList = []
        returnValue = [prefix , '((' , self.type]
        if allFeat == False : 
            fs = ['<fs']
            for key in self.__attributes.keys() :
                fs.append(key + "='" + self.getAttribute(key) + "'")
            delim = ' '
            fs[-1] = fs[-1] + '>'
            
        else :
            fs = self.fsList
            delim = '|'
        
        returnStringList.append('\t'.join(x for x in returnValue) + '\t' + delim.join(x for x in fs))
        nodePosn = 0
        for node in self.nodeList :
            nodePosn += 1
            if isinstance(node,chunkNode) :
                returnStringList.extend(node.printSSFValue(prefix + '.' + str(nodePosn), allFeat))
            else :
                returnStringList.append(node.printSSFValue(prefix + '.' + str(nodePosn), allFeat))
        returnStringList.append('\t' + '))')
        return returnStringList

    def getAttribute(self,key) :
        if self.__attributes.has_key(key) :
            return self.__attributes[key]
        else :
            return None

    def addAttribute(self,key,value) :
        self.__attributes[key] = value

    def deleteAttribute(self,key) :
        del self.__attributes[key]

class tree() :
    def __init__(self) :
        self.sentence = None
        self.sentenceID = None
        self.sentenceType = None
        self.length = 0
        self.tree = None
        self.sequence = []
        self.nodeList = []
        self.edges = {}
        self.nodes = {}
        self.tokenNodes = {}
        self.rootNode = None
        self.fileName = None
        self.comment = None
        self.probSent = False
        self.errors = []

    def populateTokenNodes(self , naming = 'Manual') :
        tokenList = returnTokenList(self.nodeList)
        assignTokenNodeNames(tokenList)
        for nodeElement in tokenList :
            self.tokenNodes[nodeElement.name] = nodeElement
        
    def updateIsParent(self) :
        parentList = []
        for node in self.nodeList :
            if node.parent!=None and node.parent!='0':
                parentList.append(node.parent)
        for node in self.nodeList :
            if node.name!=None and node.name in parentList :
                node.isParent=True
                
    def populateNodes(self , naming = 'strict') :
        if naming == 'strict' : 
            for nodeElement in self.nodeList :
                assert nodeElement.name is not None
                self.nodes[nodeElement.name] = nodeElement
        else :
            for nodeElement in self.nodeList :
                assert not (nodeElement.isParent and (nodeElement.name is None))
                if nodeElement.isParent :
                    self.nodes[nodeElement.name] = nodeElement
        return 1
    
    def addNode(self,node) :
        self.nodeList.append(node)

    def populateEdges(self) :
        for node in self.nodeList :
            nodeName = node.name
            if node.parent == '0' :
                self.rootNode = node.name
                continue
            elif node.parent not in self.nodes.iterkeys() :
#                self.errors.append('Error : Bad DepRel Parent Name ' + self.fileName + ' : ' + str(self.sentenceID) + ' : ' + node.name + ' : ' + node.parent)
                return 0
            assert node.parent in self.nodes.iterkeys()
            self.addEdge(node.parent , node.name)
        return 1
        
    def addEdge(self, parent , child) :
        if parent in self.edges.iterkeys() :
            if child not in self.edges[parent] : 
                self.edges[parent].append(child)
        else :
            self.edges[parent] = [child]

    def updateAttributes(self) :
        populateNodesStatus = self.populateNodes()
        populateEdgesStatus = self.populateEdges()
        self.sentence = self.generateSentence()
        if populateEdgesStatus == 0 or populateNodesStatus == 0:
#            print self.edges , self.nodes.keys()
            return 0
        return 1

    def printValue(self,rootNode) :
        stack = []
        printStack = self.createPrintStack(rootNode,0,stack)
        spacingLen = 0
        lastNode = ['root',0]
        printText = ''
        for nodeName in printStack :
            node = self.nodes[nodeName[0]]
            spacing = ' '*(nodeName[1]*13)
            printText += spacing + '|---'  + node.printValue() + ' (' + node.parentRelation + ')' + '\n'
            lastNode = nodeName
        return printText
            
    def createPrintStack(self,rootNodeName,level,stack) :
        if rootNodeName not in self.nodes.keys() :
            return stack
        stack.append([rootNodeName,level])
        if rootNodeName in self.edges.iterkeys() :
            for childNode in self.edges[rootNodeName] :
                stack = self.createPrintStack(childNode,level+1,stack)
        return stack

    def printSSFValue(self, allFeat = True) :
        returnStringList = []
        returnStringList.append("<Sentence id='" + str(self.sentenceID) + "'>")
        if self.nodeList != [] :
            nodeList = self.nodeList
            nodePosn = 0
            for node in nodeList :
                nodePosn += 1
                returnStringList.extend(node.printSSFValue(str(nodePosn), allFeat))
        returnStringList.append( '</Sentence>\n')
        return '\n'.join(x for x in returnStringList)

    def checkTreeSanity(self) :
        rootNodeFlag = 0
        if self.edges == {} :
            return 0
        for node in self.nodes :
            if self.nodes[node].parent == None:
                return 0
            elif self.nodes[node].parent == '0' :
                if rootNodeFlag == 0 :
                    rootNode = self.nodes[node].name
                    rootNodeFlag = 1
                else :
                    return 0
            elif self.nodes[node].parent not in self.nodes.iterkeys() :
                return 0
            
        if rootNodeFlag == 0 :
            return 0
        if rootNode not in self.nodes :
            return 0
        return 1

    def isCyclic(self) :
        assert self.nodes!={} and self.edges!={}
        parentNodes = [self.rootNode]
        assert self.rootNode in self.edges
        childNodes = self.edges[self.rootNode]
        while childNodes!=[] :
            childNode = childNodes[0]
            if childNode in parentNodes :
                print 'CYCLICITY'
                return 1
            childNodes = childNodes[1:]
            childNodes.extend(self.edges.setdefault(childNode,[]))
            parentNodes.append(childNode)
        return 0

    def generateSentence(self) :
        sentence = []
        for nodeName in self.sequence :
            sentence.append(nodeName.printValue())
        return ' '.join(x for x in sentence)


def assignNodeNames(nodeList) :
    nameDict = {}
    for node in nodeList :
        count = nameDict.setdefault(node.type , 1)
        if count !=1 :
            node.addAttribute('name', node.type + str(count))
        else :
            node.addAttribute('name', node.type)
        node.assignName()
        nameDict[node.type] += 1

def assignTokenNodeNames(nodeList) :
    nameDict = {}
    for node in nodeList :
        count = nameDict.setdefault(node.lex , 1)
        if count !=1 :
            node.addAttribute('name', node.lex + str(count))
        else : 
            node.addAttribute('name', node.lex)
        node.assignName()
        nameDict[node.lex] += 1

def returnTokenList(nodeList) :
    tokenList = []
    for nodeIter in nodeList :
        if isinstance(nodeIter, chunkNode)==True :
            tokenList.extend(returnTokenList(nodeIter.nodeList))
        elif isinstance(nodeIter, node)==True:
            tokenList.append(nodeIter)
    return tokenList

def returnChunkList(nodeList) :
    tokenList = []
    for nodeIter in nodeList :
        if isinstance(nodeIter, chunkNode)==True :
            tokenList.append(nodeIter)
            tokenList.extend(returnChunkList(nodeIter.nodeList))
    return tokenList    

def getChunkFeats(lineList) :
    chunkType = None
    fsList = []
    if len(lineList) >= 3 : 
        chunkType = lineList[2]
        
    returnFeats = {}
    multipleFeatRE = r'<fs.*?>'
    featRE = r'(?:\W*)(\S+)=([\'|\"])?([^ \t\n\r\f\v\'\"]*)[\'|\"](?:.*)'
    fsList = re.findall(multipleFeatRE, ' '.join(lineList))
    for x in lineList :
        feat = re.findall(featRE, x)
        if feat!=[] :
            if len(feat) > 1 :
                returnErrors.append('Feature with more than one value')
                continue
            returnFeats[feat[0][0]] = feat[0][2]

    return [chunkType,returnFeats,fsList]

def getTokenFeats(lineList) :
    tokenType, token = None , None
    returnFeats = {}
    fsList = []
    if len(lineList) >=3 :
        tokenType = lineList[2]

    token = lineList[1]
    multipleFeatRE = r'<fs.*?>'
    featRE = r'(?:\W*)(\S+)=([\'|\"])?([^ \t\n\r\f\v\'\"]*)[\'|\"](?:.*)'
    fsList = re.findall(multipleFeatRE, ' '.join(lineList))
    for x in lineList :
        feat = re.findall(featRE, x)
        if feat!=[] :
            if len(feat) > 1 :
                returnErrors.append('Feature with more than one value')
                continue
            returnFeats[feat[0][0]] = feat[0][2]

    return [token,tokenType,returnFeats,fsList]

def processFile(inpFD , ignoreErrors=False , treeType='dict',nesting = False) :
    currentTree = tree()
    currentTree.fileName = inpFD.name
    if treeType == 'list' :
        treeList = []
    else : 
        treeList = {}
    sentenceNo = None
    sentAlignedTo = None
    sentenceRE = r'<Sentence id=[\'|\"](\d+)[\'|\"]?>'
    endSentenceRE = r'</Sentence>'
    localNodeList = {}
    chunkType = ''
    name = ''
    drel = ''
    chunkSpan = 0
    nestedChunkList = []
    for line in inpFD :
        stripLine = line.strip()
        if stripLine=="" :
            continue
        elif stripLine[0]=="<" :
            match = re.findall(sentenceRE,stripLine) 
            if match!=[] :
                ## print inpFD.name , match
                sentenceNo = int(match[0][0])
            else :
                match = re.findall(endSentenceRE,stripLine)
                if match!=[] :
                    if currentTree.probSent == 0 :
                        currentTree.sentenceID = sentenceNo
                        sentenceNo = None
                        if ignoreErrors == False : 
                            updateAttributesStatus = currentTree.updateAttributes()
                            if updateAttributesStatus == 0 :
                                currentTree.probsent = True
                                currentTree.errors.append("Cannot update the Attributes for this sentence")

                        if treeType == 'list' :
                            currentTree.sentenceID = len(treeList)+1
                            treeList.append(currentTree)
                        elif treeType == 'dict' :
                            if currentTree.sentenceID == None :
                                currentTree.sentenceID = len(treeList) + 1
                            if currentTree.sentenceID not in treeList.iterkeys() :
                                treeList[currentTree.sentenceID] = currentTree
                            else :
                                currentTree.probSent = True
                                currentTree.errors.append('Error : Multiple Trees with same ID ')
                        elif treeType == 'madeDict' :
                            currentTree.sentenceID = len(treeList)+1
                            treeList[currentTree.sentenceID] = currentTree
                            
		    else :
			if treeType == 'list' :
			    currentTree.sentenceID = len(treeList) + 1
			    treeList.append(currentTree)
			elif treeType == 'madeDict' :
			    currentTree.sentenceID = len(treeList) + 1
			    treeList[currentTree.sentenceID] = currentTree

		    currentTree = tree()
		    currentTree.fileName = inpFD.name
	else :

	    splitLine = stripLine.split()
	    if splitLine[0] == '))' :
		if chunkSpan>1 :
                    nestedChunkList[-1].updateNodeList(localNodeList[chunkSpan])
                    del localNodeList[chunkSpan]
                    chunkSpan -= 1
                    localNodeList[chunkSpan].append(nestedChunkList.pop())
		    continue
		if ignoreErrors == False :
		    if newChunkNode.name == None :
			currentTree.errors.append('Name Field not specified')
			currentTree.probSent = True
		newChunkNode.updateNodeList(localNodeList[1])
                newChunkNode.updateNodeList(nestedChunkList)
                nestedChunkList = []
		currentTree.addNode(newChunkNode)
		currentTree.sequence.append(newChunkNode)
		localNodeList = {}
                chunkSpan = 0
		
	    elif splitLine[1] == '((' :

		if chunkSpan > 0 and nesting == True :
		    chunkSpan += 1
                    localNodeList[chunkSpan] = []                
                    [chunkType,chunkFeatDict,chunkFSList] = getChunkFeats(splitLine)
                    tempChunkNode = chunkNode(chunkType,chunkFeatDict,chunkFSList)
                    nestedChunkList.append(tempChunkNode)
		    continue

    		[chunkType,chunkFeatDict,chunkFSList] = getChunkFeats(splitLine)
    		newChunkNode = chunkNode(chunkType,chunkFeatDict,chunkFSList)

		chunkSpan = 1
                localNodeList[chunkSpan] = []                
		if chunkFeatDict.has_key('comment') :
		    currentTree.probSent = True
		    currentTree.errors.append('Comment of Probsent')
	    else :
		tokenFeats = getTokenFeats(splitLine)
		newNode = node(tokenFeats[0],tokenFeats[1],tokenFeats[2],tokenFeats[3])
		if newNode.errors!=[] :
		    currentTree.errors.extend(newNode.errors)
		    currentTree.probSent = True
		if chunkSpan > 0 :
		    localNodeList[chunkSpan].append(newNode)
		elif chunkSpan == 0 :
		    currentTree.sequence.append(newNode)
    return treeList

def folderWalk(folderPath):
    import os
    fileList = []
    for dirPath , dirNames , fileNames in os.walk(folderPath) :
        for fileName in fileNames : 
            fileList.append(os.path.join(dirPath , fileName))
    return fileList

if __name__ == '__main__' :
    
    inputPath = sys.argv[1]
    fileList = folderWalk(inputPath)
    totalSent,errorSent = 0, 0
    newFileList = []
    for fileName in fileList :
        xFileName = fileName.split('/')[-1]
        if xFileName == 'err.txt' or xFileName.split('.')[-1] in ['comments','bak'] or xFileName[:4] == 'task' :
            continue
        else :
            newFileList.append(fileName)

    for fileName in newFileList :
        inputFD = codecs.open(fileName , 'r' , encoding='utf-8')
        treeList = processFile(inputFD, treeType='madeDict',nesting=True)
        totalSent += len(treeList)
        for treeIter in treeList.iterkeys() :
            myTree = treeList[treeIter]
            myTree.sentence = myTree.generateSentence()
#            print fileName , treeIter , myTree.sentence
#            print myTree.printSSFValue()
            if myTree.errors!=[] :
                errorSent += 1
        inputFD.close()
    print totalSent , errorSent
    
