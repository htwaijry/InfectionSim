# -*- coding: utf-8 -*-

import time
import ipywidgets as widgets
from IPython.display import display

import numpy as np
from scipy.spatial import distance_matrix

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

_SIM_IS_NOTEBOOK = False

def init_notebook():
    global _SIM_IS_NOTEBOOK
    output_notebook()
    _SIM_IS_NOTEBOOK = True

class Person:
    def __init__(self, location, speed=0.005, infected=False, immunity=0.2,
                 contagious=0.8, infectRadius=0.02, resolveProb=0.01,
                 maxInfectTick=500, redirectProb=0.25, destination=np.array([0.5, 0.5]),
                 destinationTicks=20, goDestination=False, destinationProb=0.005,
                doDistance=False, minDistance=0.03, doQuarantine=False, quarantineTick=6,
                symptomProb=0.2):
        assert isinstance(location, np.ndarray), 'location is not an array'
        assert 0 <= location[0] <= 1, 'invaild x {}'.format(location[0])
        assert 0 <= location[1] <= 1, 'invaild y {}'.format(location[1])
        assert isinstance(infected, bool)
        assert 0 <= immunity <= 1, 'invaild immunity {}'.format(immunity)
        assert 0 <= contagious <= 1, 'invalid contagious {}'.format(contagious)
        assert 0 < infectRadius <= 0.2, 'invalid infectRadius {}'.format(infectRadius)
        assert 0 <= resolveProb <= 1, 'invalid resolveProb {}'.format(resolveProb)
        assert 0 < maxInfectTick, 'invalid maxInfectTick {}'.format(maxInfectTick)
        assert 0 <= redirectProb <= 1, 'invalid redirectProb {}'.format(redirectProb)
        assert isinstance(destination, np.ndarray) and destination.ndim == 1 and destination.shape[0] == 2
        assert 0 < destinationTicks, 'invalid destinationTicks {}'.format(destinationTicks)
        assert isinstance(goDestination, bool)
        assert 0 <= destinationProb <= 1, 'invalid destinationProb {}'.format(destinationProb)
        assert isinstance(doQuarantine, bool)
        assert 0 < quarantineTick, 'invalid quarantineTick {}'.format(quarantineTick)
        assert 0 <= symptomProb <= 1, 'invalid symptomProb {}'.format(symptomProb)

        self.speed = speed
        self.infected = infected
        self.location = location
        self.immunity = immunity
        self.contagious = contagious
        self.infectRadius = infectRadius
        self.resolved = False
        self.resolveProb = resolveProb
        self.maxInfectTick = maxInfectTick
        self.t = 0
        self.infectTick = 0
        self.direction = np.zeros((2,))
        self.redirectProb = redirectProb
        self.countInfected = 0
        self.goDestination = goDestination
        self.destinationProb = destinationProb
        self.destinationTicks = destinationTicks
        self.destination = destination + ((np.random.rand(2)*2)-1)*0.01
        self.isGoing = False
        self.goingTick = 0
        self.prevLocation = None
        self.doDistance = doDistance
        self.minDistance = minDistance
        self.doQuarantine = doQuarantine
        self.quarantined = False
        self.willQuarantine = np.random.rand() > symptomProb
        self.quarantineTick = quarantineTick
        

    def _setResolved(self):
        self.resolved = True
        self.infected = False
        self.immunity = 1.0
        self.quarantined = False

    def tick(self, closestPt=np.array([-1, -1]), closestDist=1):
        if self.infected:
            if self.infectTick >= self.maxInfectTick:
                self._setResolved()
            else:
            # attempt resolve
                if np.random.rand() < self.resolveProb:
                    self._setResolved()

                    
            if self.doQuarantine and not self.quarantined and not self.resolved and \
                self.willQuarantine and self.infectTick >= self.quarantineTick:
                self.quarantined = True
                    
            self.infectTick += 1
            
            
        if not self.quarantined:
            self._move(closestPt=closestPt, closestDist=closestDist)

        self.t +=1

    def _move(self, closestPt=np.array([-1, -1]), closestDist=1):
        if self.goDestination and not self.isGoing and np.random.rand() < self.destinationProb:
            self.isGoing = True
            self.goingTick = 0
            self.prevLocation = self.location.copy()
        elif not self.isGoing:
            if self.doDistance and closestDist < self.minDistance: # run away!
                self.direction = self.location - closestPt
                norm = np.linalg.norm(self.direction)
                self.direction /= norm
            elif np.random.rand() < self.redirectProb:
                self.direction = np.random.rand(2)*2-1
                norm = np.linalg.norm(self.direction)
                self.direction /= norm

            speed = self.speed
            if self.doDistance:
                speed /= 10 # also decrease speed    

            self.location += self.direction * speed
            
        elif self.isGoing:
            if self.goingTick < self.destinationTicks:
                diff = (self.destination - self.location)/(self.destinationTicks-self.goingTick)
                self.location += diff
            else:
                diff = (self.prevLocation - self.location)/(2*self.destinationTicks-self.goingTick)
                self.location += diff
                if self.goingTick == 2*self.destinationTicks-1:
                    self.isGoing = False
                    
            self.goingTick += 1
        else:
            print('Unhandled move condition!')
            
        np.clip(self.location, 0, 1, out=self.location)
    
    def infect(self, target, distance):
        assert isinstance(target, Person), 'target is not a person'
        assert 0 <= distance <= 1, 'invalid distance {}'.format(distance)

        if not self.infected or target.infected or target.resolved:
            print('Not infected or target is already infected!')
            return False
  
        if distance < self.infectRadius:
            infectProb = self.contagious * (1.0-target.immunity)
            if np.random.rand() < infectProb:
                target.infected = True
                self.countInfected += 1
                return True
    
        return False

def animate(simData, nPersons, nFrames, sleep=0.05):
    assert len(simData) > 0
    assert nPersons > 0
    assert nFrames > 0
    global fig1, fig2
    global _SIM_IS_NOTEBOOK
    fig1 = figure(plot_height=400, plot_width=400, x_range=(0,1), y_range=(0,1),
           background_fill_color='#efefef')
    
    locs, infectMask, resolveMask, quarantineMask = simData[0]
    availMask = np.logical_not(np.logical_or(infectMask, resolveMask))
    
    avail = fig1.scatter(locs[availMask][:,0], locs[availMask][:,1], marker='o', size=5,
                      fill_color='blue', alpha=0.75)
    infected = fig1.scatter(locs[infectMask][:,0], locs[infectMask][:,1], marker='o', size=5,
                      fill_color='red', alpha=0.75)
    recovered = fig1.scatter(locs[resolveMask][:,0], locs[resolveMask][:,1], marker='o', size=5,
                      fill_color='green', alpha=0.75)
    quarantined = fig1.scatter(locs[quarantineMask][:,0], locs[quarantineMask][:,1], marker='o', size=5,
                      fill_color='yellow', alpha=0.75)


    countsAvailable = np.zeros((nFrames))
    countsInfected = np.zeros((nFrames))
    countsResolved = np.zeros((nFrames))

    countsAvailable[0] = availMask.sum()
    countsInfected[0] = infectMask.sum()
    countsResolved[0] = resolveMask.sum()

    stackedSource = ColumnDataSource(data={
                                           'x':np.arange(nFrames),
                                           'available':countsAvailable,
                                           'infections':countsInfected,
                                           'resolved':countsResolved,
                                           })

    fig2 = figure(plot_width=400, plot_height=400, x_range=(0, nFrames), y_range=(0, nPersons),)

    areaStacks = fig2.varea_stack(['infections', 'available','resolved'], x='x', 
                                 color=("red", "blue", "green"), source=stackedSource)

    p = row(fig1, fig2)
    
    show(p, notebook_handle=True)
    
    for i, (locs, infectMask, resolveMask, quarantineMask) in enumerate(simData):
        availMask = np.logical_not(np.logical_or(infectMask, resolveMask))
        avail.data_source.data = {'x': locs[availMask][:,0], 'y':locs[availMask][:,1]}
        infected.data_source.data = {'x':locs[infectMask][:,0], 'y':locs[infectMask][:,1]}
        recovered.data_source.data = {'x':locs[resolveMask][:,0], 'y':locs[resolveMask][:,1]}
        quarantined.data_source.data = {'x':locs[quarantineMask][:,0], 'y':locs[quarantineMask][:,1]}

        nInfected = infectMask.sum()
        nResolved = resolveMask.sum()
        countsAvailable[i] = nPersons-(nInfected+nResolved)
        countsInfected[i] = nInfected
        countsResolved[i] = nResolved

        stackedData={
           'x':np.arange(nFrames),
           'available':countsAvailable,
           'infections':countsInfected,
           'resolved':countsResolved
           }

        for areaStack in areaStacks:
            areaStack.data_source.data = stackedData

        time.sleep(sleep)
        if _SIM_IS_NOTEBOOK:
            push_notebook()
		
def advance(persons, personLocations, infectedMask, resolvedMask, quarantinedMask, infectRadius):
    distMat = distance_matrix(personLocations, personLocations) 
    closestIdx = np.argsort(distMat, axis=1)[:,1]
    for idx, person in enumerate(persons):
        person.tick(closestPt=personLocations[closestIdx[idx]], closestDist=distMat[idx, closestIdx[idx]])
        if person.infected:
            infectedMask[idx] = True
            if person.quarantined:
                quarantinedMask[idx] = True
        elif person.resolved:
            infectedMask[idx] = False
            quarantinedMask[idx] = False
            resolvedMask[idx] = True

    if infectedMask.any():
        infectedIdx = np.arange(len(persons))[infectedMask]
        infectedLocations = personLocations[infectedIdx]
        # attempt infect
        distMat = distance_matrix(infectedLocations, personLocations) 
        infectTgt = distMat < infectRadius
        for idx, tgtRow in enumerate(infectTgt):
            tgtIdxs = np.arange(len(persons))[tgtRow]
            srcIdx = infectedIdx[idx]
            src = persons[srcIdx]
            if not src.quarantined:
                for tgtIdx in tgtIdxs:
                    if tgtIdx == srcIdx:
                        continue
                    tgt = persons[tgtIdx]
                    if tgt.infected or tgt.resolved:
                        continue
                    if src.infect(tgt, distMat[idx, tgtIdx]):
                        infectedMask[tgtIdx] = True

    
    assert not np.logical_and(infectedMask, resolvedMask).any(), 'infected and resolved!'

    return personLocations.copy(), infectedMask.copy(), resolvedMask.copy(), quarantinedMask.copy()
	
def simulate(nFrames=500, nPersons=200, nInfected=3, infectRadius=0.05,
             immunity=0.2, immunityRange=0.1,
             contagious=0.8, contagiousRange=0.1,
             resolveProb=0.01, maxInfectTicks=500, travelTicks=20, 
             doTravel=False, travelProb=0.005, socialDistance=False, 
             minDistance=0.03, socialDistancePct=1.0, socialDistanceActThr=0,
             doQuarantine=False, quarantineTicks=6, quarantineActThr=0, symptomaticProb=0.2):
    
    personLocations = np.random.uniform(size=(nPersons, 2))
    
    persons = [Person(location, 
                      immunity=np.random.uniform(immunity-immunity*immunityRange, immunity+immunity*immunityRange),
                      contagious=np.random.uniform(contagious-contagious*contagiousRange, contagious+contagious*contagiousRange),
                      infectRadius=infectRadius, resolveProb=resolveProb,
                      maxInfectTick=maxInfectTicks,
                      destinationTicks=travelTicks, goDestination=doTravel, destinationProb=travelProb,
                      doDistance=False, minDistance=minDistance,
                      doQuarantine=False, quarantineTick=quarantineTicks, symptomProb=symptomaticProb) 
           for location in personLocations
          ]       
    
    infectedIdx = np.random.permutation(np.arange(nPersons))[:nInfected]
    for idx in infectedIdx:
        persons[idx].infected = True

    infectedMask = np.zeros((nPersons), dtype=np.bool)
    resolvedMask = np.zeros((nPersons), dtype=np.bool)
    quarantinedMask = np.zeros((nPersons), dtype=np.bool)

    infectedMask[infectedIdx] = True

    simData = []

    needSocialDistance = socialDistance and socialDistanceActThr > 0
    didActSocialDist = False
    needQuarantine = doQuarantine and quarantineActThr > 0
    didActQuarantine = False
    for i in range(nFrames):
        if needSocialDistance and not didActSocialDist:
            if infectedMask.sum() > socialDistanceActThr:
                didActSocialDist = True
                for person in persons:
                    if np.random.rand() < socialDistancePct:
                        person.doDistance = True
        if needQuarantine and not didActQuarantine:
            if infectedMask.sum() > quarantineActThr:
                didActQuarantine = True
                for person in persons:
                    person.doQuarantine=True
            
        simData.append(advance(persons, personLocations, infectedMask, resolvedMask, quarantinedMask, infectRadius))
        
    return simData
	
def showMenu():
    global runConfig
    global simData
    runConfig = dict(nFrames=1000, nPersons=300, nInfected=3, infectRadius=0.03,
                     immunity=0.2, immunityRange=0.1,           
                     contagious=0.8, contagiousRange=0.1,
                     resolveProb=0.01, maxInfectTicks=500, travelTicks=20, 
                     doTravel=False, travelProb=0.005, socialDistance=False, 
                     minDistance=0.03, socialDistancePct=1.0, socialDistanceActThr=1,
                     doQuarantine=False, quarantineTicks=6, quarantineActThr=1, symptomaticProb=0.2)
    simData = None
    
    def handleChange(change):
        global runConfig
        cfgKey = change['owner'].cfgKey
        runConfig[cfgKey] = change['new']

    def runClicked(button):
        global runConfig
        global simData
        global arabicNames
        button.disabled=True
        print("جاري حساب المحاكاه")
        valStr = ''
        for i, key in enumerate(runConfig):
            val = runConfig[key]
            if isinstance(val, bool):
                val = 'نعم' if val else 'لا'
            valStr += '{}: {} ## '.format(arabicNames[key], val)
            if (i+1) % 4 == 0:
                valStr += '\n'
        print(valStr)
        simData = simulate(**runConfig)
        print("جاري تشغيل المحاكاه")
        animate(simData, runConfig['nPersons'], runConfig['nFrames'], sleep=0.03)
        button.disabled=False
    
    def playAnimation(button):
        global runConfig
        global simData
        if simData is not None:
            button.disabled=True
            animate(simData, runConfig['nPersons'], runConfig['nFrames'], sleep=0.03)
            button.disabled=False
        else:
            print("يتوجب حساب المحاكاه أولًا")

    def makeWidget(objType, cfgKey, **kwargs):
        w = objType(style={'description_width': 'initial'}, layout=widgets.Layout(width='50%'), **kwargs)
        w.cfgKey = cfgKey
        w.observe(handleChange, names='value')
        return w

    nFrameW = makeWidget(widgets.IntSlider, 'nFrames', min=1, max=5000, value=1000, description='عدد الخطوات')
    nPersonW = makeWidget(widgets.IntSlider, 'nPersons', min=1, max=500, value=300, description='عدد الأشخاص')
    nInfectedW = makeWidget(widgets.IntSlider, 'nInfected', min=1, max=1000, value=3, description='عدد المصابين ابتداءًا')
    infectRadiusW = makeWidget(widgets.FloatSlider, 'infectRadius', min=0.01, step=0.01, max=0.1, value=0.03, description='المسافة المعديه')
    immunityW = makeWidget(widgets.FloatSlider, 'immunity', min=0.0, step=0.1, max=1.0, value=0.2, description='المناعة ضد المرض')
    immunityRangeW = makeWidget(widgets.FloatSlider, 'immunityRange', min=0.05, step=0.05, max=1.0, value=0.1, description='تباين المناعة')
    contagiousW = makeWidget(widgets.FloatSlider, 'contagious', min=0.0, step=0.1, max=1.0, value=0.8, description='احتمال وقوع العدوى')
    contagiousRangeW = makeWidget(widgets.FloatSlider, 'contagiousRange', min=0.01, step=0.01, max=0.1, value=0.05, description='تباين العدوى')
    resolveProbW = makeWidget(widgets.FloatSlider, 'resolveProb', min=0.0, step=0.005, max=0.5, value=0.01, readout_format='.3f', description='احتمال التشافي')
    maxInfectTicksW = makeWidget(widgets.IntSlider, 'maxInfectTicks', min=1, max=1000, value=500, description='سقف خطوات الاصابه للشخص')
    doTravelW = makeWidget(widgets.Checkbox, 'doTravel', value=False, description='تفعيل نقطة للتجمع')
    travelTicksW = makeWidget(widgets.IntSlider, 'travelTicks', min=1, max=50, value=20, description='عدد خطوات الانتقال لنقطة تجمع')
    travelProbW = makeWidget(widgets.FloatSlider, 'travelProb', min=0.0, step=0.001, max=0.1, value=0.005, readout_format='.3f', description='احتمال الذهاب لنقطة التجمع')
    socialDistanceW = makeWidget(widgets.Checkbox, 'socialDistance', value=False, description='تفعيل تجنب الآخرين')
    minDistanceW = makeWidget(widgets.FloatSlider, 'minDistance', min=0.01, step=0.005, max=0.1, value=0.03, readout_format='.3f', description='مسافة التجنب')
    socialDistancePctW = makeWidget(widgets.FloatSlider, 'socialDistancePct', min=0.0, step=0.1, max=1.0, value=1.0, description='نسبة الاشخاص المتبعين لنظام التجنب')
    socialDistanceActThrW = makeWidget(widgets.IntSlider, 'socialDistanceActThr', min=1, max=1000, value=20, description='تفعيل تجنب الآخرين بعد اكتشاف عدد حالات')

    doQuarantineW = makeWidget(widgets.Checkbox, 'doQuarantine', value=False, description='تفعيل الحجر الصحي')
    quarantineTicksW = makeWidget(widgets.IntSlider, 'quarantineTicks', min=1, max=50, value=6, description='عدد الخطوات قبل اكتشاف المرض')
    symptomaticProbW = makeWidget(widgets.FloatSlider, 'symptomaticProb', min=0.0, step=0.1, max=1.0, value=0.2, description='احتمال المرض بدون اعراض')
    quarantineActThrW = makeWidget(widgets.IntSlider, 'quarantineActThr', min=1, max=1000, value=20, description='تفعيل الحجر الصحي بعد اكتشاف عدد حالات')

    runButton = widgets.Button(
        description='محاكاة',
        disabled=False,
        button_style='',
        icon=''
    )
    runButton.on_click(runClicked)

    playButton = widgets.Button(
        description='تشغيل آخر محاكاة',
        disabled=False,
        button_style='',
        icon=''
    )
    playButton.on_click(playAnimation)
    
    allWidgets = [nFrameW, nPersonW, nInfectedW, infectRadiusW, immunityW, immunityRangeW, 
                  contagiousW, contagiousRangeW, resolveProbW, maxInfectTicksW, doTravelW, 
                  travelTicksW,travelProbW,socialDistanceW, minDistanceW, socialDistancePctW,
                  socialDistanceActThrW, doQuarantineW, quarantineTicksW,symptomaticProbW,quarantineActThrW]
    
    # sync default vals
    for w in allWidgets:
        w.value = runConfig[w.cfgKey]
        
    global arabicNames
    arabicNames = {w.cfgKey:w.description for w in allWidgets}
    
    display(*allWidgets)
    display(runButton, playButton)