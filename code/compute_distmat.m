base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion_new'; 

anat_dir = fullfile(base_dir, 'MDTB', 'derivatives', 'ffimport');

resultsPath = fullfile('/Users/jkderrick028/Documents/Projects/7TfMRI/pfc_parcellation/results/START_A7_spatialACF_map_LH/MDTB'); 

sn = [2, 3, 4];
subjects = compose('sub-%02d', sn); 

n_subjects = numel(subjects); 

dist_matrices = 0;
maxradius = 45; 

for subj = 1:n_subjects
    disp("Now processing " + subjects(subj))
    white = fullfile(anat_dir, subjects{subj}, 'anat', ...
    [subjects{subj}, '_space-32k_hemi-L_white.surf.gii']);

    pial = fullfile(anat_dir, subjects{subj}, 'anat', ...
    [subjects{subj}, '_space-32k_hemi-L_pial.surf.gii']);

    for s=sn
        C1=gifti(white);
        C2=gifti(pial);
        vertices = double((C1.vertices' + C2.vertices' )/2); % Midgray surface
        faces    = double(C1.faces');
        numVert=size(vertices,2);
        n2f=surfing_nodeidxs2faceidxs(faces);
        D=inf([numVert numVert],'single');
        for i=1:numVert
            [subV, subF, subIndx,vidxs, fidxs]=surfing_subsurface(vertices, faces, i, maxradius, n2f); % construct correct sub surface
            D(vidxs,i)=single(surfing_dijkstradist(subV,subF,subIndx,maxradius));            
        end
        dist_matrices = dist_matrices + D; 
    end  
end

dist_matrices = dist_matrices / n_subjects; 

save(fullfile(resultsPath, 'distanceMat_matlab_version_MDTB.mat'), "dist_matrices", '-v7.3'); 






% function varargout=sc1_sc2_neocortical(what,varargin)
% % Analysis of cortical data
% % This is modernized from the analysis in sc1_sc2_imana, in that it uses
% % the new FS_LR template
% 
% % baseDir    = '/Volumes/MotorControl/data/super_cerebellum_new';
% baseDir    = 'Z:\data\super_cerebellum_new';
% wbDir      = fullfile(baseDir,'sc1','surfaceWB');
% fsDir      = fullfile(baseDir,'sc1','surfaceFreesurfer');
% atlasDir   = '~/Data/Atlas_templates/standard_mesh';
% anatomicalDir = fullfile(baseDir,'sc1','anatomicals');
% regDir     = fullfile(baseDir,'sc1','RegionOfInterest');
% glmDir      = 'GLM_firstlevel_4';
% studyDir  = {'sc1','sc2'};
% Hem       = {'L','R'};
% hemname   = {'CortexLeft','CortexRight'};
% fshem     = {'lh','rh'};
% subj_name = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',...
%     's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24',...
%     's25','s26','s27','s28','s29','s30','s31'};
% returnSubjs=[2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31];
% 
% switch(what)
% 
% case 'DCBC:computeDistances' % Compute individual Dijkstra distances between vertices
%     sn=returnSubjs;
%     hem = [1 2];
%     resolution = '32k';
%     maxradius = 40;
% 
%     vararginoptions(varargin,{'sn','hem','resolution','maxradius'});
%     for s=sn
%         for h=hem
%             surfDir = fullfile(wbDir, subj_name{s});
%             white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',subj_name{s},Hem{h},resolution));
%             pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',subj_name{s},Hem{h},resolution));
%             C1=gifti(white);
%             C2=gifti(pial);
%             vertices = double((C1.vertices' + C2.vertices' )/2); % Midgray surface
%             faces    = double(C1.faces');
%             numVert=size(vertices,2);
%             n2f=surfing_nodeidxs2faceidxs(faces);
%             D=inf([numVert numVert],'single');
%             for i=1:numVert;
%                 [subV, subF, subIndx,vidxs, fidxs]=surfing_subsurface(vertices, faces, i, maxradius, n2f); % construct correct sub surface
%                 D(vidxs,i)=single(surfing_dijkstradist(subV,subF,subIndx,maxradius));
%                 if (mod(i,100)==0)
%                     fprintf('.');
%                 end
%             end;
%             fprintf('\n');
%             save(fullfile(surfDir,sprintf('distances.%s.mat',Hem{h})),'D','-v7.3');
%         end;
%     end;
% case 'DCBC:avrgdistances'    % Average individual distances
%     sn=[2,3,4,6,8,9,10,12,14];
%     hem = [1 2];
%     resolution = '32k';
%     vararginoptions(varargin,{'sn','hem','resolution','maxradius'});
%     avrgD=single(zeros(32492,32492));
%     N=length(sn);
%     for h=hem
%         for s=1:length(sn)
%             fprintf('.');
%             surfDir = fullfile(wbDir, subj_name{sn(s)});
%             load(fullfile(surfDir,sprintf('distances.%s.mat',Hem{h})));
%             D(isinf(D))=45;
%             w=single((s-1)/N);
%             avrgD = w.*avrgD+(1-w)*D;
%         end;
%         fprintf('\n');
%         avrgDs=sparse(double(avrgD));
%         save(fullfile(wbDir,'group32k',sprintf('distances_sp.%s.mat',Hem{h})),'avgrDs');
%     end;