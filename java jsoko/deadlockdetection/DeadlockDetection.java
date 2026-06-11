/**
 *  JSoko - A Java implementation of the game of Sokoban
 *  Copyright (c) 2012 by Matthias Meger, Germany
 *
 *  This file is part of JSoko.
 *
 *	JSoko is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package de.sokoban_online.jsoko.deadlockdetection;

import de.sokoban_online.jsoko.board.Board;
import de.sokoban_online.jsoko.resourceHandling.Settings;
import de.sokoban_online.jsoko.resourceHandling.Settings.SearchDirection;
import de.sokoban_online.jsoko.utilities.Debug;


/**
 * This class checks the board for a deadlock.
 */
public final class DeadlockDetection {

	// For easy access we use a direct reference to the board.
	final Board board;

	/** Object for detecting corral deadlocks. */
	public final CorralDeadlockDetection corralDeadlockDetection;

	/** Object for detecting freeze deadlocks. */
	public final FreezeDeadlockDetection freezeDeadlockDetection;

	/** Object for detecting closed diagonal deadlocks. */
	public final ClosedDiagonalDeadlock closedDiagonalDeadlockDetection;

	/** Object for detecting bipartite matching deadlocks. */
	public final BipartiteMatchings bipartiteDeadlockDetection;


	/**
	 * Creates an object for the deadlock detection.
	 *
	 * @param board  the board of the current level
	 */
	public DeadlockDetection(Board board) {

		this.board = board;

		// Create the objects for the deadlock detection.
		corralDeadlockDetection 		= new CorralDeadlockDetection(board, this);
		freezeDeadlockDetection 		= new FreezeDeadlockDetection(board);
		closedDiagonalDeadlockDetection = new ClosedDiagonalDeadlock(board);
		bipartiteDeadlockDetection 		= new BipartiteMatchings(board);

	}

	/**
	 * Searches for deadlock pattern on the board.<br>
	 * If a deadlock is recognized, the board is definitely unsolvable.<br>
	 * <p>
	 * Note: if no deadlock is recognized this just means that JSoko
	 * can't detect any deadlock but the board nevertheless may contain
	 * an unknown deadlock pattern.
	 *
	 * @return <code>true</code> when the current board is not solvable anymore, <code>false</code> otherwise
	 */
	public boolean isDeadlock() {
		return isDeadlock(board.boxData.getBoxPosition(0));
	}


	/**
	 * Searches for deadlock pattern on the board.<br>
	 * If a deadlock is recognized, the board is definitely unsolvable.<br>
	 * <p>
	 * Note: if no deadlock is recognized this just means that JSoko
	 * can't detect any deadlock but the board nevertheless may contain
	 * an unknown deadlock pattern.
	 *
	 * @param newBoxPosition  new position of the pushed box
	 * @return <code>true</code> when the current board is not solvable anymore, <code>false</code> otherwise
	 */
	public boolean isDeadlock(int newBoxPosition) {

		/**
		 * Detection of 'Simple deadlocks'.
		 */
		if (Settings.detectSimpleDeadlocks && board.isSimpleDeadlockSquare(newBoxPosition)) {
			return true;
		}

		long timeLimit = System.currentTimeMillis() + 100; // 100 milli seconds for the deadlock detection

		/**
		 * Detection of 'Immovable box on non-goal' deadlocks.
		 */
		if (Settings.detectFreezeDeadlocks) {

			// All frozen boxes are marked as "frozen". Checking all boxes takes some time.
			// Hence, it's assumed that the user hasn't continued playing when a freeze deadlock has
			// occurred earlier. This means we only check those frozen boxes which don't create a
			// deadlock situation that is: boxes on goals. Nevertheless, the freeze check will also
			// mark those boxes as frozen which are not on a goal that constitute in the "frozen" status
			// of a box on a goal. This means only frozen boxes on a non goal that don't contribute
			// to a "frozen" status of a "box on a goal" aren't marked as frozen here, like these two:
			// #######
			// # $$  #
			// #     #
			// ~~~~~~~
			board.boxData.setAllBoxesNotFrozen();
			for (int boxNo = 0; boxNo < board.boxCount; boxNo++) {
				int boxPosition = board.boxData.getBoxPosition(boxNo);
				if (board.isBoxOnGoal(boxPosition) && !board.boxData.isBoxFrozen(boxNo)) {
					if (freezeDeadlockDetection.isDeadlock(boxPosition, true)) {
						return true;
					}
				}
			}

			// Now check the freeze and deadlock status of the pushed box. This needn't to be done
			// if the box is on a goal, since in that case the check would have already been done above.
			if (!board.isBoxOnGoal(newBoxPosition) && freezeDeadlockDetection.isDeadlock(newBoxPosition, true)) {
				return true;
			}
		}

		/**
		 * Detection of 'Closed diagonal' deadlocks.
		 */
		if(Settings.detectClosedDiagonalDeadlocks && closedDiagonalDeadlockDetection.isDeadlock(newBoxPosition)) {
			return true;
		}

		/**
		 * Detection of "bipartite matching" deadlocks.
		 * This deadlock detection leverages the found "frozen" box information.
		 */
		if (Settings.detectBipartiteDeadlocks) {
			if(bipartiteDeadlockDetection.isDeadlock(SearchDirection.FORWARD)) {
				return true;
			}
		}

		long timeToStopCorralDetection = Debug.debugCorral ? timeLimit + 100000 : timeLimit;

		/**
		 * Detection of 'Corral' deadlocks. At least 10 ms should be left to do a detection.
		 */
		if (Settings.detectCorralDeadlocks && (timeToStopCorralDetection - System.currentTimeMillis() > 10)) {
			if (corralDeadlockDetection.isDeadlock(newBoxPosition, timeToStopCorralDetection)) {
				return true;
			}
		}

		// No deadlock has been found.
		return false;
	}

	/**
	 * Searches for deadlock pattern on the board regarding pulls instead of pushes.<br>
	 * If a deadlock is recognized, the board is definitely unsolvable.<br>
	 * <p>
	 * Note: if no deadlock is recognized this just means that JSoko
	 * can't detect any deadlock but the board nevertheless may contain
	 * an unknown deadlock pattern.
	 *
	 * @return <code>true</code> when the current board is not solvable anymore, <code>false</code> otherwise
	 */
	public boolean isBackwardDeadlock() {
		return isBackwardDeadlock(board.boxData.getBoxPosition(0));
	}

	/**
	 * Searches for deadlock pattern on the board regarding pulls instead of pushes.<br>
	 * If a deadlock is recognized, the board is definitely unsolvable.<br>
	 * <p>
	 * Note: if no deadlock is recognized this just means that JSoko
	 * can't detect any deadlock but the board nevertheless may contain
	 * an unknown deadlock pattern.
	 *
	 * @param newBoxPosition  new position of the pushed box
	 * @return <code>true</code> when the current board is not solvable anymore, <code>false</code> otherwise
	 */
	public boolean isBackwardDeadlock(int newBoxPosition) {

		// Simple deadlock squares are valid for pulls and for pushes.
		if (board.isSimpleDeadlockSquare(newBoxPosition)) {
			return true;
		}

		if(bipartiteDeadlockDetection.isDeadlock(SearchDirection.BACKWARD)) {
			return true;
		}

		return false;
	}
}