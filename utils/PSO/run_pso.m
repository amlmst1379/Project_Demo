nVar = 3;
VarMin = [0, 0, 0];        % Lower bounds for [alpha, phi1, phi2]
VarMax = [pi, pi, pi];     % Upper bounds

nPop = 30;     % number of particles
MaxIt = 50;    % iterations
w = 0.7;       % inertia
c1 = 1.5;      % cognitive (self)
c2 = 1.5;      % social (swarm)

for i = 1:nPop
    particle(i).Position = VarMin + rand(1,nVar).*(VarMax - VarMin);
    particle(i).Velocity = zeros(1,nVar);
    particle(i).Cost     = DAB_Objective(particle(i).Position);
    particle(i).Best = particle(i);
end

[~, idx] = min([particle.Cost]);
GlobalBest = particle(idx).Best;

for it = 1:MaxIt
    for i = 1:nPop
        r1 = rand(1,nVar);
        r2 = rand(1,nVar);
        
        particle(i).Velocity = ...
            w * particle(i).Velocity ...
          + c1 * r1 .* (particle(i).Best.Position - particle(i).Position) ...
          + c2 * r2 .* (GlobalBest.Position - particle(i).Position);
        
        particle(i).Position = particle(i).Position + particle(i).Velocity;

        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);
        
        particle(i).Cost = DAB_Objective(particle(i).Position);

        if particle(i).Cost < particle(i).Best.Cost
            particle(i).Best = particle(i);
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
    end

    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(GlobalBest.Cost)]);
end