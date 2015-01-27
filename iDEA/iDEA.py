# Library imports
import parameters as pm
import iDEA_MB2
import iDEA_MB3
import Density
import sprint

# Set message level
msglvl = pm.msglvl

# Print splash
sprint.sprint('                                                              ',1,0,msglvl)
sprint.sprint('                  *    ****     *****       *                 ',1,0,msglvl)
sprint.sprint('                       *   *    *          * *                ',1,0,msglvl)
sprint.sprint('                  *    *    *   *         *   *               ',1,0,msglvl)
sprint.sprint('                  *    *     *  *****    *     *              ',1,0,msglvl)
sprint.sprint('                  *    *    *   *       *********             ',1,0,msglvl)
sprint.sprint('                  *    *   *    *      *         *            ',1,0,msglvl)
sprint.sprint('                  *    ****     ***** *           *           ',1,0,msglvl)
sprint.sprint('                                                              ',1,0,msglvl)
sprint.sprint('  ------------------------------------------------------------',1,0,msglvl)
sprint.sprint('  |           Interacting Dynamic Electrons Approach         |',1,0,msglvl)
sprint.sprint('  |              to Many-body Quantum Mechanics              |',1,0,msglvl)
sprint.sprint('  |                                                          |',1,0,msglvl)
sprint.sprint('  |                 Created by Piers Lillystone,             |',1,0,msglvl)                         
sprint.sprint('  |                 Matt Hodgson, Jacob Chapman,             |',1,0,msglvl)
sprint.sprint('  |               Thomas Durrant & Jack Wetherell            |',1,0,msglvl)
sprint.sprint('  |                   The University of York                 |',1,0,msglvl)
sprint.sprint('  ------------------------------------------------------------',1,0,msglvl)
sprint.sprint('                                                              ',1,0,msglvl)
sprint.sprint('                                                              ',2,0,msglvl)
sprint.sprint('                                                              ',2,0,msglvl)
sprint.sprint('                  *    ****     *****       *                 ',2,0,msglvl)
sprint.sprint('                       *   *    *          * *                ',2,0,msglvl)
sprint.sprint('                  *    *    *   *         *   *               ',2,0,msglvl)
sprint.sprint('                  *    *     *  *****    *     *              ',2,0,msglvl)
sprint.sprint('                  *    *    *   *       *********             ',2,0,msglvl)
sprint.sprint('                  *    *   *    *      *         *            ',2,0,msglvl)
sprint.sprint('                  *    ****     ***** *           *           ',2,0,msglvl)
sprint.sprint('                                                              ',2,0,msglvl)
sprint.sprint('  ------------------------------------------------------------',2,0,msglvl)
sprint.sprint('  |           Interacting Dynamic Electrons Approach         |',2,0,msglvl)
sprint.sprint('  |              to Many-body Quantum Mechanics              |',2,0,msglvl)
sprint.sprint('  |                                                          |',2,0,msglvl)
sprint.sprint('  |                 Created by Piers Lillystone,             |',2,0,msglvl)                         
sprint.sprint('  |                 Matt Hodgson, Jacob Chapman,             |',2,0,msglvl)
sprint.sprint('  |               Thomas Durrant & Jack Wetherell            |',2,0,msglvl)
sprint.sprint('  |                   The University of York                 |',2,0,msglvl)
sprint.sprint('  ------------------------------------------------------------',2,0,msglvl)
sprint.sprint('                                                              ',2,0,msglvl)

# Run required jobs
if(pm.NE == 2):
    iDEA_MB2.main()
    Density.Run()
    if(pm.TDDFT == 1):
	import iDEA_TDDFT2
if(pm.NE == 3):
    iDEA_MB3.main()
    if(pm.TDDFT == 1):
	import iDEA_TDDFT3

# All jobs done
sprint.sprint('All Jobs Done.',1,0,msglvl)
sprint.sprint('All Jobs Done.',2,0,msglvl)
