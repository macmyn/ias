//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

#include <AztecOO.h>

#include "aux.h"


int main(int argc, char **argv)
{
    using namespace Tensor;
    using namespace std;
    using namespace ias;
    using Teuchos::RCP;
    using Teuchos::rcp;
        
    MPI_Init(&argc, &argv);
    
    //---------------------------------------------------------------------------
    // [0] Input parameters
    //---------------------------------------------------------------------------
    int nx{4};
    int ny{1};
    int nz{1};
    double dx{2.1};
    double dy{2.1};
    double dz{2.1};
    double R{1.0};
    int nSubdiv{3};
    
    bool restart{false};
    string resLocation;
    string resFileName;

    double     intEL = 1.E-1;
    double     intCL = 5.E-2;
    double     intSt = 1.0;
    double   tension = 1.E-1;
    double     kappa = 1.E-2;
    double viscosity = 1.E0;
    double frictiont = 1.E-1;
    double frictionn = 1.E-1;
    
    double lifetime = 1.0;

    double totTime{1.E3};
    double deltat{1.E-2};
    double stepFac{0.95};
    double maxDeltat{1.0};

    int    nr_maxite{5};
    double nr_restol{1.E-8};
    double nr_soltol{1.E-8};
        
    string  fEnerName{"energies.txt"};
    string  updateMethod{"eulerian"};
    ofstream fEner;
    
    if(argc == 2)
    {
        const char *config_filename = argv[1];
        ConfigFile config(config_filename);

        config.readInto(       nx, "nx");
        config.readInto(       ny, "ny");
        config.readInto(       nz, "nz");
        
        config.readInto(       dx, "dx");
        config.readInto(       dy, "dy");
        config.readInto(       dz, "dz");

        config.readInto(        R, "R");

        config.readInto(  nSubdiv, "nSubdiv");

        config.readInto(    intEL, "intEL");
        config.readInto(    intCL, "intCL");
        config.readInto(    intSt, "intSt");
        config.readInto(  tension, "tension");
        config.readInto(    kappa, "kappa");
        config.readInto(viscosity, "viscosity");
        config.readInto(frictiont, "frictiont");
        config.readInto(frictionn, "frictionn");
        
        config.readInto( lifetime, "lifetime");

        config.readInto(  totTime,   "totTime");
        config.readInto(   deltat,   "deltat");
        config.readInto(  stepFac,   "stepFac");
        config.readInto(  maxDeltat, "maxDeltat");

        config.readInto(nr_maxite, "nr_maxite");
        config.readInto(nr_restol, "nr_restol");
        config.readInto(nr_soltol, "nr_soltol");

        config.readInto(fEnerName, "fEnerName");
        
        config.readInto(restart, "restart");
        config.readInto(resLocation, "resLocation");
        config.readInto(resFileName, "resFileName");
        
        config.readInto(updateMethod, "updateMethod");
        
        if(updateMethod.compare("eulerian")!=0 and updateMethod.compare("ale") != 0)
            throw runtime_error("Update method not understood. It should be either \"eulerian\" or \"ALE\"");
    }
    //---------------------------------------------------------------------------
    
    RCP<Tissue> tissue;
    if(!restart)
    {
        RCP<TissueGen> tissueGen = rcp( new TissueGen);
        tissueGen->setBasisFunctionType(BasisFunctionType::LoopSubdivision);
        
        tissueGen->addNodeFields({"vx","vy","vz"});
        tissueGen->addNodeFields({"x0","y0","z0"});
        tissueGen->addNodeFields({"vx0","vy0","vz0"});

        tissueGen->addCellFields({"P","Paux","P0","Paux0"});
        tissueGen->addCellFields({"A","X","Y","Z"});
        tissueGen->addCellFields({"A0","X0","Y0","Z0"});
        tissueGen->addCellFields({"Ai"});

        tissueGen->addCellFields({"intEL","intCL","intSt","tension","kappa","viscosity","frictiont","frictionn"});
        
        tissueGen->addTissField("time");
        tissueGen->addTissField("deltat");

        tissue = tissueGen->genRegularGridSpheres(nx, ny, nz, dx, dy, dz, R, nSubdiv);
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        tissue->balanceDistribution();
        
        tissue->calculateInteractingElements(intEL+3.0*intCL);
        
        tissue->getTissField("time") = 0.0;
        tissue->getTissField("deltat") = deltat;
        
        for(auto cell: tissue->getLocalCells())
        {
            cell->getCellField("intEL") = intEL;
            cell->getCellField("intCL") = intCL;
            cell->getCellField("intSt") = intSt;
            cell->getCellField("kappa") = kappa;
            cell->getCellField("tension") = tension;
            cell->getCellField("viscosity") = viscosity;
            cell->getCellField("frictiont") = frictiont;
            cell->getCellField("frictionn") = frictionn;
        }
        tissue->saveVTK("Cell","_t"+to_string(0));
    }
    else
    {
        tissue = rcp(new Tissue);
        tissue->loadVTK(resLocation, resFileName, BasisFunctionType::LoopSubdivision);
        tissue->Update();
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->balanceDistribution();
        tissue->updateGhosts();
        tissue->calculateInteractingElements(intEL+3.0*intCL);
        
        deltat = tissue->getTissField("deltat");
        
        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("vx") *= deltat;
            cell->getNodeField("vy") *= deltat;
            cell->getNodeField("vz") *= deltat;
        }
    }
        
    tissue->saveVTK("Cell","_t"+to_string(1));

    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({"vx","vy","vz"});
    physicsIntegration->setCellDOFs({"P"});
    physicsIntegration->setSingleIntegrand(internal);
    physicsIntegration->setDoubleIntegrand(interaction);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    physicsIntegration->setCellIntegralFields({"A","X","Y","Z","A0","X0","Y0","Z0","Ai"});
    physicsIntegration->setCellDOFsInInteractions(false);
    physicsIntegration->Update();

    RCP<Integration> eulerianIntegration = rcp(new Integration);
    eulerianIntegration->setTissue(tissue);
    eulerianIntegration->setNodeDOFs({"x","y","z"});
    eulerianIntegration->setCellDOFs({"Paux"});
    if(updateMethod.compare("eulerian")==0)
        eulerianIntegration->setSingleIntegrand(eulerianUpdate);
    else
        eulerianIntegration->setSingleIntegrand(arbLagEulUpdate);
    eulerianIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    eulerianIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    eulerianIntegration->Update();

    RCP<solvers::TrilinosAztecOO> physicsLinearSolver = rcp(new solvers::TrilinosAztecOO);
    physicsLinearSolver->setIntegration(physicsIntegration);
    physicsLinearSolver->addAztecOOParameter("solver","gmres");
    physicsLinearSolver->addAztecOOParameter("precond","dom_decomp");
    physicsLinearSolver->addAztecOOParameter("subdomain_solve","ilu");
    physicsLinearSolver->addAztecOOParameter("output","none");
    physicsLinearSolver->setMaximumNumberOfIterations(5000);
    physicsLinearSolver->setResidueTolerance(1.E-8);
    physicsLinearSolver->Update();
    
    RCP<solvers::TrilinosAztecOO> eulerianLinearSolver = rcp(new solvers::TrilinosAztecOO);
    eulerianLinearSolver->setIntegration(eulerianIntegration);
    eulerianLinearSolver->addAztecOOParameter("solver","gmres");
    eulerianLinearSolver->addAztecOOParameter("precond","dom_decomp");
    eulerianLinearSolver->addAztecOOParameter("subdomain_solve","ilu");
    eulerianLinearSolver->addAztecOOParameter("output","none");
    eulerianLinearSolver->setMaximumNumberOfIterations(500);
    eulerianLinearSolver->setResidueTolerance(1.E-8);
    eulerianLinearSolver->Update();
    
    RCP<solvers::NewtonRaphson> physicsNewtonRaphson = rcp(new solvers::NewtonRaphson);
    physicsNewtonRaphson->setLinearSolver(physicsLinearSolver);
    physicsNewtonRaphson->setSolutionTolerance(1.E-8);
    physicsNewtonRaphson->setResidueTolerance(1.E-8);
    physicsNewtonRaphson->setMaximumNumberOfIterations(5);
    physicsNewtonRaphson->setVerbosity(true);
    physicsNewtonRaphson->Update();
    
    RCP<solvers::NewtonRaphson> eulerianNewtonRaphson = rcp(new solvers::NewtonRaphson);
    eulerianNewtonRaphson->setLinearSolver(eulerianLinearSolver);
    eulerianNewtonRaphson->setSolutionTolerance(1.E-8);
    eulerianNewtonRaphson->setResidueTolerance(1.E-8);
    eulerianNewtonRaphson->setMaximumNumberOfIterations(5);
    eulerianNewtonRaphson->setVerbosity(true);
    eulerianNewtonRaphson->Update();
    
    int step{};
    double time = tissue->getTissField("time");

    fEner.open (fEnerName);
    fEner.close();
    
    int conv{};
    bool rec_str{};
    while(time < totTime)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("x0")  = cell->getNodeField("x");
            cell->getNodeField("y0")  = cell->getNodeField("y");
            cell->getNodeField("z0")  = cell->getNodeField("z");
            cell->getNodeField("vx0") = cell->getNodeField("vx");
            cell->getNodeField("vy0") = cell->getNodeField("vy");
            cell->getNodeField("vz0") = cell->getNodeField("vz");
            
            cell->getCellField("P0")    = cell->getCellField("P");
            cell->getCellField("Paux0") = cell->getCellField("Paux");
        }
        tissue->updateGhosts();
        
        auto finish_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_1 = finish_1 - start;

        if(conv)
            rec_str = max(rec_str,tissue->calculateInteractingElements(intEL+3.0*intCL));
        else if(rec_str)
        {
            physicsIntegration->recalculateMatrixStructure();
            physicsLinearSolver->DestroyPreconditioner();
            rec_str = false;
        }
        
        auto finish_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_2 = finish_2 - finish_1;
        
        if(tissue->getMyPart()==0)
            cout << "Step " << step << ", time=" << time << ", deltat=" << deltat << endl;

        
        if(tissue->getMyPart()==0)
            cout << "Solving for velocities" << endl;
        
        physicsNewtonRaphson->solve();
        conv = physicsNewtonRaphson->getConvergence();
        
        auto finish_3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_3 = finish_3 - finish_2;
        
        
        if ( conv )
        {
            int nIter = physicsNewtonRaphson->getNumberOfIterations();
            
            if(tissue->getMyPart()==0)
                cout << "Solving for displacement along the normal" << endl;

            eulerianNewtonRaphson->solve();
            conv = eulerianNewtonRaphson->getConvergence();
            
            if (conv)
            {
                if(updateMethod.compare("ale")==0)
                {
                    for(auto cell: tissue->getLocalCells())
                    {
                        cell->getNodeField("x") += cell->getCellField("X")/cell->getCellField("A") - cell->getCellField("X0")/cell->getCellField("A0");
                        cell->getNodeField("y") += cell->getCellField("Y")/cell->getCellField("A") - cell->getCellField("Y0")/cell->getCellField("A0");
                        cell->getNodeField("z") += cell->getCellField("Z")/cell->getCellField("A") - cell->getCellField("Z0")/cell->getCellField("A0");
                    }
                }
                
                time += deltat;
                tissue->getTissField("time") = time;

                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("vx") /= deltat;
                    cell->getNodeField("vy") /= deltat;
                    cell->getNodeField("vz") /= deltat;
                }
                tissue->saveVTK("Cell","_t"+to_string(step+1));
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("vx") *= deltat;
                    cell->getNodeField("vy") *= deltat;
                    cell->getNodeField("vz") *= deltat;
                }

//                if(tissue->getMyPart() == 0)
//                {
//                    fEner.open (fEnerName,ios::app);
//                    fEner << setprecision(8) << scientific;
//                    fEner << time << " " << deltat << " " << physicsIntegration->getGlobalIntegral(0) << " " << physicsIntegration->getGlobalIntegral(1) << " " << physicsIntegration->getGlobalIntegral(2) << " " << physicsIntegration->getGlobalIntegral(3) << endl;
//                    fEner.close();
//                }

                if(nIter < nr_maxite)
                {
                    deltat /= stepFac;
                    for(auto cell: tissue->getLocalCells())
                    {
                        cell->getNodeField("vx") /= stepFac;
                        cell->getNodeField("vy") /= stepFac;
                        cell->getNodeField("vz") /= stepFac;
                    }
                }
                if(deltat > maxDeltat)
                    deltat = maxDeltat;
                step++;
            }
            else
            {
                cout << "CAREFUL!!! Solver for avxiliary field failed!" << endl;
                deltat *= stepFac;
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("x") = cell->getNodeField("x0");
                    cell->getNodeField("y") = cell->getNodeField("y0");
                    cell->getNodeField("z") = cell->getNodeField("z0");
                    cell->getCellField("Paux") = cell->getCellField("Paux0");

                }
                tissue->updateGhosts();
            }
        }
        else
        {
            deltat *= stepFac;
            
            for(auto cell: tissue->getLocalCells())
            {
                cell->getNodeField("vx") = cell->getNodeField("vx0") * stepFac;
                cell->getNodeField("vy") = cell->getNodeField("vy0") * stepFac;
                cell->getNodeField("vz") = cell->getNodeField("vz0") * stepFac;
                cell->getCellField("P")  = cell->getCellField("P0");
            }
            tissue->updateGhosts();
        }

        auto finish_4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_4 = finish_4 - finish_3;
        
        tissue->getTissField("deltat") = deltat;
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        auto finish_5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_5 = finish_5 - finish_4;

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        
        if(tissue->getMyPart()==0)
        {
            cout << "Duration of time-step: " << elapsed.count() << endl;
            cout << "    "<< "Update ref: " << elapsed_1.count() << endl;
            cout << "    "<< "Calculate int elements: " << elapsed_2.count() << endl;
            cout << "    "<< "Newton Raphson for v: " << elapsed_3.count() << endl;
            cout << "    "<< "Newton Raphson for x: " << elapsed_4.count() << endl;
            cout << "    "<< "Update cell adjacency: " << elapsed_5.count() << endl;
        }
    }
    
//    tissue->closePVD();
    
    MPI_Finalize();

    return 0;
}
