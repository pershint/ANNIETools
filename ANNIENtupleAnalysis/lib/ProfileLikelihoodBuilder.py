import numpy as np
import scipy as sp

class ProfileLikelihoodBuilder(object):
    '''
    Class is used to build likelihood profiles given a signal and background
    hypothesis.  For this class, the signal is assumed to follow a Bernoulli distribution
    while the background is a Poisson distribution with some mean.
    '''
    def __init__(self):
        self.likelihood_function = None
        self.bkg_pois_mean = None
        self.bkg_pois_mean_unc = None

    def SetBkgMean(self,mean):
        self.bkg_pois_mean = mean

    def SetBkgMeanUnc(self,mean_unc):
        self.bkg_pois_mean_unc = mean_unc

    def BuildLikelihoodProfile(self,ProfileVariable,SignalDistribution,SignalDistribution_unc,randShoots,BackgroundDistribution,BkgDistUnc = None):
        '''
        Returns a Chi-squared profile given the input profile variables, the background mean neutron rate
        defined with the SetBkgMean method, and a signal distribution.
        '''
        if self.bkg_pois_mean is None:
            print("You have to set your background distribution's mean neutron count per window!")
            return None
        ChiSquares = []
        LowestChiSq = 1E12
        LowestChiSqProfile = None
        for var in ProfileVariable:
            #MCProfile = self.BuildMCProfile(var,SignalDistribution,randShoots)
            MCProfile = self.BuildMCProfileBkgDist(var,SignalDistribution,BackgroundDistribution,randShoots,BkgDistUnc)
            ChiSquare = self.CalcChiSquare(MCProfile,SignalDistribution,SignalDistribution_unc,BkgDistUnc)
            if ChiSquare<LowestChiSq:
                LowestChiSqProfile = MCProfile
                LowestChiSq = ChiSquare
            ChiSquares.append(ChiSquare)
        return ChiSquares, LowestChiSqProfile

    def BuildMCProfile(self,variable,SignalDistribution,randShoots):
        '''
        Given the probability of returning a 1 in the signal distribution and the defined background mean in
        self.bkg_pois_mean, build a Monte Carlo-based data distribution.
        '''
        Profile = np.zeros(len(SignalDistribution))
        poisson_means = np.random.normal(self.bkg_pois_mean,self.bkg_pois_mean_unc,randShoots)
        poisson_means = poisson_means[np.where(poisson_means>0)[0]]
        randShoots = len(poisson_means)
        neutron_counts = np.zeros(randShoots)
        randos = np.random.random(randShoots)
        saw_neutrons = np.where(randos<=variable)[0]
        neutron_counts[saw_neutrons]+=1
        Bkg_shoots = np.random.poisson(poisson_means)
        #Bkg_shoots = np.random.poisson(self.bkg_pois_mean,randShoots)
        neutron_counts = neutron_counts + Bkg_shoots
        Profile,Profile_edges = np.histogram(neutron_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def BuildMCProfileBkgDist(self,variable,SignalDistribution,BkgDistribution,randShoots,BkgDistUnc=None):
        '''
        Given the probability of returning a 1 in the signal distribution and the defined background mean in
        self.bkg_pois_mean, build a Monte Carlo-based data distribution.
        '''
        Profile = np.zeros(len(SignalDistribution))
        Bin_values = np.arange(0,len(SignalDistribution),1)
        neutron_counts = np.zeros(int(randShoots))
        neutron_counts += np.random.choice(Bin_values, int(randShoots),p=BkgDistribution)
        randos = np.random.random(int(randShoots))
        saw_neutrons = np.where(randos<=variable)[0]
        neutron_counts[saw_neutrons]+=1
        Profile,Profile_edges = np.histogram(neutron_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def CalcChiSquare(self,MCProfile,SignalDistribution,SignalDistribution_unc,BkgDistUnc):
        '''
        Given a hypothesis profile (MCProfile), calculate the chi-square relative to the
        input signal distribution.
        '''
        return np.sum(((MCProfile-SignalDistribution)/np.sqrt(SignalDistribution_unc**2 + BkgDistUnc**2))**2)



class ProfileLikelihoodBuilder2D(object):
    '''
    Class is used to build likelihood profiles given a signal and background
    hypothesis.  For this class, the signal is assumed to follow a Bernoulli distribution
    while the background is a Poisson distribution with some mean.
    '''
    def __init__(self,ndim = 3):
        self.likelihood_function = None
        self.xprofile = None
        self.yprofile = None

    def SetEffProfile(self,xpro):
        self.xprofile = xpro

    def SetBkgMeanProfile(self,ypro):
        self.yprofile = ypro


    def BuildLikelihoodProfile(self,SignalDistribution,SignalDistribution_unc,randShoots):
        '''
        Returns a Chi-squared profile given the input profile variables, the background mean neutron rate
        defined with the SetBkgMean method, and a signal distribution.
        '''
        if self.xprofile is None or self.yprofile is None:
            print("Please define xprofile variables and yprofile variables")
            return None
        x_variable = []
        y_variable = []
        ChiSquares = []
        LowestChiSq = 1E12
        LowestChiSqProfile = None
        for xvar in self.xprofile:
            for yvar in self.yprofile:
                x_variable.append(xvar)
                y_variable.append(yvar)
                MCProfile = self.BuildMCProfile(xvar,yvar,SignalDistribution,randShoots)
                ChiSquare = self.CalcChiSquare(MCProfile,SignalDistribution,SignalDistribution_unc)
                if ChiSquare<LowestChiSq:
                    LowestChiSqProfile = MCProfile
                    LowestChiSq = ChiSquare
                ChiSquares.append(ChiSquare)
        return np.array(x_variable),np.array(y_variable),np.array(ChiSquares), LowestChiSqProfile

    def BuildMCProfile(self,neu_eff,bkg_rate,SignalDistribution,randShoots):
        '''
        Given the neutron capture efficiency and mean neutron multiplicity of the
        uncorrelated bkg component, build a Monte Carlo-based data distribution.
        '''
        neutron_counts = np.zeros(int(randShoots))
        randos = np.random.random(int(randShoots))
        saw_neutrons = np.where(randos<=neu_eff)[0]
        neutron_counts[saw_neutrons]+=1
        Bkg_shoots = np.random.poisson(bkg_rate,int(randShoots))
        #Bkg_shoots = np.random.poisson(self.bkg_pois_mean,randShoots)
        neutron_counts = neutron_counts + Bkg_shoots
        Profile,Profile_edges = np.histogram(neutron_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def CalcChiSquare(self,MCProfile,SignalDistribution,SignalDistribution_unc):
        '''
        Given a hypothesis profile (MCProfile), calculate the chi-square relative to the
        input signal distribution.
        '''
        return np.sum(((MCProfile-SignalDistribution)/SignalDistribution_unc)**2)


class ProfileLikelihoodBuilder3D(object):
    '''
    Class is used to build likelihood profiles given a signal and background
    hypothesis.  For this class, the signal is assumed to follow a Bernoulli distribution
    while the background is a Poisson distribution with some mean.
    '''
    def __init__(self):
        self.xprofile = None
        self.yprofile = None
        self.zprofile = None
        #self.source_rate = 200 #Hz
        self.gamma_prob = 0.58
        self.window_size = 65E-6 #seconds
        self.capture_time = 30E-6 #seconds

    def SetNeuEffProfile(self,pro):
        self.xprofile = pro

    def SetGammaEffProfile(self,pro):
        self.yprofile = pro

    def SetBkgMeanProfile(self,pro):
        self.zprofile = pro

    def SetSourceRate(self,rate):
        self.source_rate = rate

    def BuildLikelihoodProfile(self,SignalDistribution,SignalDistribution_unc,randShoots):
        '''
        Returns a Chi-squared profile given the input profile variables, the background mean neutron rate
        defined with the SetBkgMean method, and a signal distribution.
        '''
        if len(self.xprofile) == 0 or len(self.yprofile) == 0 or len(self.zprofile) == 0:
            print("All three profile dimensions not fille with something..")

        ProfileValues = np.array(np.meshgrid(self.xprofile,self.yprofile,self.zprofile)).T
        print("PROFILE VAL LEN: " + str(ProfileValues.size))
        ProfileValues = ProfileValues.reshape(int(ProfileValues.size/3), 3)
        LowestChiSq = 1E12
        LowestChiSqProfile = None
        ChiSquares = []
        for PV in ProfileValues:
                print("CALCULATING CHISQ FOR PROFILE VALS: " + str(PV))
                MCProfile = self.BuildMCProfile(PV,SignalDistribution,randShoots)
                ChiSquare = self.CalcChiSquare(MCProfile,SignalDistribution,SignalDistribution_unc)
                if ChiSquare<LowestChiSq:
                    LowestChiSqProfile = MCProfile
                    LowestChiSq = ChiSquare
                ChiSquares.append(ChiSquare)
        return ProfileValues[0:,0],ProfileValues[0:,1],ProfileValues[0:,2], np.array(ChiSquares), LowestChiSqProfile

    def BuildMCProfile(self,ProfileValues,SignalDistribution,randShoots):
        '''
        Given the neutron capture efficiency, gamma detection efficiency, mean multiplicity of the
        uncorrelated bkg component, mean multiplicity of uncorrelated neutron component, 
        and mean multiplicity of uncorrelated gamma-neutron component, 
        build a Monte Carlo-based data distribution.
        '''
        neu_eff = ProfileValues[0]
        gamma_eff = ProfileValues[1]
        #bkg_rate = ProfileValues[2] #ORIGINAL
        source_rate = ProfileValues[2]
        #Get trigger-correlated neutron multiplicity
        neutron_counts = np.zeros(int(randShoots))
        neutroneff_randos = np.random.random(int(randShoots))
        saw_neutrons = np.where(neutroneff_randos<=neu_eff)[0]
        neutron_counts[saw_neutrons]+=1

        #Get uncorrelated background component
        #bkg_counts = np.random.poisson(bkg_rate,int(randShoots)) #ORIGINAL

        #Get uncorrelated neutron count
        #mean_n_mult = self.source_rate * (1-self.gamma_prob) * self.window_size  #ORIGINAL
        mean_n_mult = source_rate * (1-self.gamma_prob) * self.window_size 
        uncorr_neutron_counts = np.random.poisson(mean_n_mult*neu_eff,int(randShoots))

        #Get uncorrelated gamma-neutron count
        uncorr_gamman_counts = self.CalculateGammaNMultiplicity(gamma_eff,neu_eff,randShoots)

        #cluster_counts = neutron_counts  + uncorr_neutron_counts + uncorr_gamman_counts + bkg_counts #ORIGINAL
        cluster_counts = neutron_counts  + uncorr_neutron_counts + uncorr_gamman_counts #+ bkg_counts
        Profile,Profile_edges = np.histogram(cluster_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def CalculateGammaNMultiplicity(self,gamma_eff,neu_eff,randShoots):
        mean_gamma_mult = self.source_rate * (self.gamma_prob) * self.window_size
        gamma_counts = np.random.poisson(mean_gamma_mult,int(randShoots))
        obs_neutron_counts = np.zeros(int(randShoots))
        obs_gamma_counts = np.zeros(int(randShoots))
        #Shoot the number of neutrons that would have been observed, then correct for gamma eff
        nonzero_inds = np.where(gamma_counts>0)[0]
        for ind in nonzero_inds:
            x = gamma_counts[ind]
            #Shoot the times at which gammas WOULD have been seen.
            gamma_times = np.random.random(x)*self.window_size
            #Shoot neutron capture times
            n_timeshots = np.random.random(x)
            n_times_after_gamma = -(self.capture_time)*np.log(1-n_timeshots)
            neutron_times = gamma_times + n_times_after_gamma
            observed_neutrons = len(np.where(neutron_times<=self.window_size)[0])
            obs_neutron_counts[ind]+=observed_neutrons
            #Correct gamma number observed based on gamma_efficiency
            obs_gamma_counts[ind]+= len(np.where(np.random.random(x)<gamma_eff)[0])
        return obs_neutron_counts + obs_gamma_counts


    def CalcChiSquare(self,MCProfile,SignalDistribution,SignalDistribution_unc):
        '''
        Given a hypothesis profile (MCProfile), calculate the chi-square relative to the
        input signal distribution.
        '''
        return np.sum(((MCProfile-SignalDistribution)/SignalDistribution_unc)**2)

